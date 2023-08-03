import argparse
import time, os
import numpy as np
import torch, cv2
from tqdm import tqdm
from glob import glob
from PIL import Image
from torchvision import transforms

def load_model(teacher_model_checkpoint,
                student_model_checkpoint,
                autoencoder_model_checkpoint,
                teacher_meanvalue_checkpoint,
                teacher_stdvalue_checkpoint,
                device='cpu'):
    
    if "pth" in teacher_model_checkpoint or "pt" in teacher_model_checkpoint:
        device = torch.device('cuda:0')
        teacher_net = torch.load(teacher_model_checkpoint, map_location=device)
        student_net = torch.load(student_model_checkpoint, map_location=device)
        ae_net = torch.load(autoencoder_model_checkpoint, map_location=device)
        teacher_mean_tensor = torch.load(teacher_meanvalue_checkpoint, map_location=device)
        teacher_std_tensor = torch.load(teacher_stdvalue_checkpoint, map_location=device)

        teacher_net.eval(), student_net.eval(), ae_net.eval()
        return teacher_net, student_net, ae_net, teacher_mean_tensor, teacher_std_tensor

    elif "onnx" in teacher_model_checkpoint:
        import onnxruntime
        teacher_net = onnxruntime.InferenceSession(teacher_model_checkpoint)
        student_net = onnxruntime.InferenceSession(student_model_checkpoint)
        ae_net = onnxruntime.InferenceSession(autoencoder_model_checkpoint)
        teacher_mean_tensor = torch.load(teacher_meanvalue_checkpoint)
        teacher_std_tensor = torch.load(teacher_stdvalue_checkpoint)
        teacher_mean_arr = teacher_mean_tensor.detach().cpu().numpy()
        teacher_std_arr = teacher_std_tensor.detach().cpu().numpy()

        return teacher_net, student_net, ae_net, teacher_mean_arr, teacher_std_arr

@torch.no_grad()
def inference(pil_image, teacher_model, student_model, ae_model, 
                teacher_mean, teacher_std, out_channels=384,
                q_st_start=None, q_st_end=None, 
                q_ae_start=None, q_ae_end=None, 
                device='cuda:0'):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Transform for sending to model
    pil_tensor         = default_transform(pil_img)
    pil_tensor         = pil_tensor[None]
    pil_tensor         = pil_tensor.to(device)

    teacher_output     = teacher_model(pil_tensor)
    teacher_output     = (teacher_output - teacher_mean) / teacher_std
    student_output     = student_model(pil_tensor) # [1, 384, 56, 56]
    autoencoder_output = ae_model(pil_tensor)      # [1, 384, 56, 56]

    map_st = torch.mean((teacher_output - student_output[:, :out_channels]) ** 2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, out_channels:]) ** 2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae

    return map_combined, map_st, map_ae


def inference_onnx(pil_image, teacher_model, student_model, ae_model, 
                teacher_mean, teacher_std, out_channels=384,
                q_st_start=None, q_st_end=None, 
                q_ae_start=None, q_ae_end=None,
                device='cpu'):
    pil_tensor = default_transform(pil_img)
    pil_tensor = pil_tensor[None]
    arr_image = np.asarray(pil_tensor).astype(np.float32)

    ort_input = {teacher_model.get_inputs()[0].name: arr_image}
    teacher_output = teacher_model.run(None, ort_input)[0]

    teacher_output = (teacher_output - teacher_mean) / teacher_std
    del ort_input

    ort_input = {student_model.get_inputs()[0].name: arr_image}
    student_output = student_model.run(None, ort_input)[0]
    del ort_input

    ort_input = {ae_model.get_inputs()[0].name: arr_image}
    ae_output = ae_model.run(None, ort_input)[0]
    del ort_input

    map_st = np.mean((teacher_output - student_output[:, :out_channels]) ** 2, axis=1)
    map_ae = np.mean((ae_output - student_output[:, :out_channels]) ** 2, axis=1)

    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)

    map_combined = 0.5 * map_st + 0.5 * map_ae

    return map_combined, map_st, map_ae

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Inference EfficientAD")
    parser.add_argument("-d", "--data_path", default="./datasets/MVTec", help="Parent working directory of target object")
    parser.add_argument("-ckpt", "--checkpoint_path", default="./output/1/trainings/mvtec_ad", help="Parent checkpoint directory of target object")
    parser.add_argument("-im", "--inference_mode", default="pth", choices=["pth", "onnx"], help="Select PTH or ONNX mode for inference")
    parser.add_argument("-obj", "--object", default="leather", help="Select object for inference")
    parser.add_argument("-p", "--phase", default="test", choices=["train", "test"], help="Select phase folder for inference [train, test]")
    parser.add_argument("-f", "--fold", default="scratch", help="Select defect type folder for inference")
    parser.add_argument("-o", "--output_path", default="./output/1/visualization", help="Define output directory")

    config = parser.parse_args()

    # Define data path
    obj = config.object
    phase = config.phase
    fold = config.fold
    inference_mode = config.inference_mode
    checkpoint_path = config.checkpoint_path
    data_dir = f"{config.data_path}/{obj}/{phase}/{fold}"
    output_dir = f"{config.output_path}/{obj}/{phase}/{fold}"
    os.makedirs(output_dir, exist_ok=True)
    img_path_list = glob(f"{data_dir}/*.png")

    # Define input size and tensor transform
    image_size = 256
    default_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])

    if inference_mode == "pth":
        # Load the PTH model
        teacher_model_checkpoint     = f"{config.checkpoint_path}/{config.object}/teacher_final.pth"
        student_model_checkpoint     = f"{config.checkpoint_path}/{config.object}/student_final.pth"
        autoencoder_model_checkpoint = f"{config.checkpoint_path}/{config.object}/autoencoder_final.pth"
        teacher_meanvalue_checkpoint = f"{config.checkpoint_path}/{config.object}/teacher_mean.pth"
        teacher_stdvalue_checkpoint  = f"{config.checkpoint_path}/{config.object}/teacher_std.pth"

    elif inference_mode == "onnx":
        # Load the ONNX model
        teacher_model_checkpoint     = f"{config.checkpoint_path}/{config.object}/teacher_final.onnx"
        student_model_checkpoint     = f"{config.checkpoint_path}/{config.object}/student_final.onnx"
        autoencoder_model_checkpoint = f"{config.checkpoint_path}/{config.object}/autoencoder_final.onnx"
        teacher_meanvalue_checkpoint = f"{config.checkpoint_path}/{config.object}/teacher_mean.pth"
        teacher_stdvalue_checkpoint  = f"{config.checkpoint_path}/{config.object}/teacher_std.pth"

    print("[INFO]... Loading model ...")
    teacher_net, student_net, ae_net, teacher_mean_tensor, teacher_std_tensor = load_model(
        teacher_model_checkpoint,
        student_model_checkpoint,
        autoencoder_model_checkpoint,
        teacher_meanvalue_checkpoint,
        teacher_stdvalue_checkpoint
    )

    print("[INFO]... Starting inference ...")
    time_cost_list = []
    with torch.no_grad():
        for i in tqdm(range(len(img_path_list))):
            print("Processing image:\t", os.path.basename(img_path_list[i]))
            s1 = time.time()
            img_path = img_path_list[i]

            # Read input image
            pil_img = Image.open(img_path)
            orig_width = pil_img.width
            orig_height = pil_img.height

            if inference_mode == "pth":
                map_combined, map_st, map_ae = inference(pil_img, 
                                                        teacher_net, student_net, ae_net, 
                                                        teacher_mean_tensor, teacher_std_tensor,
                                                        q_st_start=None, q_st_end=None, 
                                                        q_ae_start=None, q_ae_end=None)

            elif inference_mode == "onnx":
                map_combined, map_st, map_ae = inference_onnx(pil_img,
                                                            teacher_net, student_net, ae_net, 
                                                            teacher_mean_tensor, teacher_std_tensor,
                                                            q_st_start=None, q_st_end=None, 
                                                            q_ae_start=None, q_ae_end=None)
                map_combined = torch.from_numpy(map_combined).unsqueeze(0)

            map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
            map_combined = torch.nn.functional.interpolate(
                map_combined, (orig_height, orig_width), mode='bilinear')
            map_combined = map_combined[0, 0].cpu().numpy()

            map_combined = cv2.normalize(map_combined, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            heatmap_combined = cv2.applyColorMap(map_combined, None, cv2.COLORMAP_JET)
            out = np.float32(heatmap_combined)/255 + np.float32(np.asarray(pil_img))/255
            out = out / np.max(out)
            out = np.uint8(out * 255.0)
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

            cv2.imwrite(f"{output_dir}/{os.path.basename(img_path_list[i])}", out)
            
            s2 = time.time()
            time_cost_list.append(s2 - s1)
    print("[INFO]... Finish inference! ...")
    print(f'\n[INFO]... Average time cost:\t{np.mean(time_cost_list):.6f}s ...')
