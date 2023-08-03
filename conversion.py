import os
import torch
import argparse
import numpy as np
import onnx, onnxruntime


def convert_to_onnx_dynamic(model, input_size, onnx_path):
    model.eval()

    dummy_input = torch.randn(1, *input_size, requires_grad=True)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: 'batch_size', 2: "img_height", 3: "img_width"},
            "output": {0: 'batch_size', 2: "img_height", 3: "img_width"},
        }
    )
    print("Finised converting to ONNX")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Conversion PTH to ONNX for EfficientAD")
    parser.add_argument("-ckpt", "--checkpoint_path", default="./output/1/trainings/mvtec_ad", help="Parent checkpoint directory of target object")
    parser.add_argument("-obj", "--object", default="pcb", help="Select object for inference")
    parser.add_argument("-o", "--output_path", default="./output/1/export/mvtec_ad", help="Define output directory")
    parser.add_argument("-d", "--device", default="cpu", help="Define device for conversion mode")

    config = parser.parse_args()

    out_channels = 384
    input_size = (3, 256, 256)
    obj = config.object
    output_path = config.output_path
    checkpoint_path = config.checkpoint_path
    device = torch.device(config.device)

    ### Define the PTH model path
    teacher_model_checkpoint     = f"{checkpoint_path}/{obj}/teacher_final.pth"
    student_model_checkpoint     = f"{checkpoint_path}/{obj}/student_final.pth"
    autoencoder_model_checkpoint = f"{checkpoint_path}/{obj}/autoencoder_final.pth"
    teacher_meanvalue_checkpoint = f"{checkpoint_path}/{obj}/teacher_mean.pth"
    teacher_stdvalue_checkpoint  = f"{checkpoint_path}/{obj}/teacher_std.pth"

    ### Define output ONNX model path
    onnx_output_path = f"{output_path}/{obj}"
    os.makedirs(onnx_output_path, exist_ok=True)
    teacher_onnx_checkpoint     = f"{onnx_output_path}/teacher_final.onnx"
    student_onnx_checkpoint     = f"{onnx_output_path}/student_final.onnx"
    autoencoder_onnx_checkpoint = f"{onnx_output_path}/autoencoder_final.onnx"

    ### Load the PTH model
    teacher_net = torch.load(teacher_model_checkpoint, map_location=device)
    student_net = torch.load(student_model_checkpoint, map_location=device)
    ae_net = torch.load(autoencoder_model_checkpoint, map_location=device)

    convert_to_onnx_dynamic(teacher_net, input_size, teacher_onnx_checkpoint)
    convert_to_onnx_dynamic(student_net, input_size, student_onnx_checkpoint)
    convert_to_onnx_dynamic(ae_net, input_size, autoencoder_onnx_checkpoint)
