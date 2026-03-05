#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
import onnx
from argparse import ArgumentParser
import sys
sys.path.append('..')
from Models.model_components.scene_seg_network import SceneSegNetwork
from Models.model_components.scene_3d_network import Scene3DNetwork
from Models.model_components.domain_seg_network import DomainSegNetwork
from Models.model_components.auto_speed_network import AutoSpeedNetwork
from Models.model_components.ego_lanes_network import EgoLanesNetwork
from Models.model_components.auto_steer_network import AutoSteerNetwork
from Models.model_components.auto_drive.auto_drive_network import AutoDriveNetwork
def main():

    # Argument parser for data root path and save path
    parser = ArgumentParser()
    parser.add_argument("-n", "--name", dest="network_name", required=True, \
                        help="specify the name of the network which will be benchmarked")

    parser.add_argument("-p", "--model_checkpoint_path", dest="model_checkpoint_path", required=True, \
                        help="path to pytorch checkpoint file to load model dict")

    parser.add_argument("-o", "--onnx_model_path", dest="onnx_model_path", required=True, \
                        help="path to converted ONNX model, must include output file name with .onnx extension")

    args = parser.parse_args()

    # Get input arguments
    model_name = args.network_name
    model_checkpoint_path = args.model_checkpoint_path
    onnx_model_path = args.onnx_model_path

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for inference')


    # Instantiating Model and setting to evaluation mode
    model = 0
    if(model_name == 'SceneSeg'):
        print('Processing SceneSeg Network')
        model = SceneSegNetwork()
    elif (model_name == 'Scene3D'):
        print('Processing Scene3D Network')
        sceneSegNetwork = SceneSegNetwork()
        model = Scene3DNetwork(sceneSegNetwork)
    elif (model_name == 'DomainSeg'):
        print('Processing DomainSeg Network')
        sceneSegNetwork = SceneSegNetwork()
        model = DomainSegNetwork(sceneSegNetwork)
    elif (model_name == 'AutoSpeed'):
        print('Processing AutoSpeed Network')
        autospeed_builder = AutoSpeedNetwork()
        model = autospeed_builder.build_model(version='n', num_classes=4)
    elif (model_name == 'EgoLanes'):
        print('Processing EgoLanes Network')
        model = EgoLanesNetwork()
    elif (model_name == 'AutoSteer'):
        print('Processing AutoSteer Network')
        model = AutoSteerNetwork()
    elif (model_name == 'AutoDrive'):
        print('Processing AutoDrive Network')
        autodrive_builder = AutoDriveNetwork()
        model = autodrive_builder.build_model(version='n', num_classes=4)
    else:
        raise Exception("Model name not specified correctly, please check")

    # Loading Pytorch checkpoint
    print('Loading Network')
    if(len(model_checkpoint_path) > 0):
            checkpoint = torch.load(model_checkpoint_path, weights_only=False, map_location=device)
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                # If checkpoint['model'] is a model object, extract its state_dict
                if hasattr(checkpoint['model'], 'state_dict'):
                    model.load_state_dict(checkpoint['model'].state_dict())
                else:
                    model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
    else:
        raise ValueError('No path to checkpiont file provided in class initialization')
    model = model.to(device)
    model = model.eval()

    # Fake input data (AutoSpeed uses 640x640)
    if model_name == 'AutoSpeed':
        input_shape=(1, 3, 640, 640)
    if model_name == 'AutoSteer':
        input_shape=(1, 6, 80, 160)
    if model_name == 'AutoDrive':
        input_shape = (1, 3, 512, 1024)
    else:
        input_shape=(1, 3, 320, 640)
    input_data = torch.randn(input_shape)
    input_data = input_data.to(device)

    # Test inference
    print('Testing inference')
    _ = model(input_data)

    # Export FP32 model to onnx
    print('Converting model to ONNX at FP32 and exporting')
    torch.onnx.export(model,                                          # model
                    input_data,                                       # model input
                    onnx_model_path,                                  # path
                    export_params=True,                               # store the trained parameter weights inside the model file
                    opset_version=14,                                 # the ONNX version to export the model to
                    do_constant_folding=True,                         # constant folding for optimization
                    input_names = ['input'],                          # input names
                    output_names = ['output'],                        # output names
                    dynamic_axes={'input' : {0 : 'batch_size'},       # variable length axes
                                    'output' : {0 : 'batch_size'}})

    # Run checks on exported FP32 ONNX network
    ONNX_network = onnx.load(onnx_model_path)
    onnx.checker.check_model(ONNX_network)
    print('Checks passed - export complete')

if __name__ == '__main__':
  main()
# %%