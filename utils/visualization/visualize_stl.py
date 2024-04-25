import os
import vtk
from medpy import io
import numpy as np


def visualize_stl(stl_path):
    filename = stl_path

    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)

    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(reader.GetOutput())
    else:
        mapper.SetInputConnection(reader.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create a rendering window and renderer
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    # Create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Assign actor to the renderer
    ren.AddActor(actor)

    # Enable user interface interactor
    iren.Initialize()
    renWin.Render()
    iren.Start()


def convert_mha_to_stl(mha_path, stl_path=None, visualize=False):

    mha_file_path = mha_path
    stl_file_path = stl_path
    if stl_file_path is None and visualize is False:
        return None

    reader = vtk.vtkMetaImageReader()
    reader.SetFileName(mha_file_path)
    reader.Update()

    extra = vtk.vtkMarchingCubes()
    extra.SetInputConnection(reader.GetOutputPort())
    extra.SetValue(0, 1)

    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(extra.GetOutputPort())

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(stripper.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create a rendering window and renderer
    # ren = vtk.vtkRenderer()
    # renWin = vtk.vtkRenderWindow()
    # renWin.AddRenderer(ren)
    # renWin.SetSize(600, 600)

    # Create a renderwindowinteractor
    # iren = vtk.vtkRenderWindowInteractor()
    # iren.SetRenderWindow(renWin)

    # Assign actor to the renderer
    # ren.AddActor(actor)

    # Enable user interface interactor
    # if visualize:
    #     iren.Initialize()
    #     renWin.Render()
    #     iren.Start()
    # if stl_file_path is None:
    #     return None

    triangle = vtk.vtkTriangleFilter()
    triangle.SetInputConnection(extra.GetOutputPort())
    triangle.PassVertsOff()
    triangle.PassLinesOff()

    decimation = vtk.vtkQuadricDecimation()
    decimation.SetInputConnection(triangle.GetOutputPort())

    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(triangle.GetOutputPort())

    triangle2 = vtk.vtkTriangleFilter()
    triangle2.SetInputConnection(clean.GetOutputPort())
    triangle2.PassVertsOff()
    triangle2.PassLinesOff()

    stlWriter = vtk.vtkSTLWriter()
    stlWriter.SetInputConnection(triangle2.GetOutputPort())
    stlWriter.SetFileName(stl_file_path)
    stlWriter.SetFileTypeToBinary()
    stlWriter.Write()

    print(stl_file_path + " is saved.")

    return None


def save_numpy_as_stl(
    np_array, save_dict, stl_name, visualize=False, spacing=(1, 1, 1)
):
    if not os.path.exists(save_dict):
        os.makedirs(save_dict)

    if stl_name[-4::] == ".stl" or stl_name[-4::] == ".mha":
        stl_name = stl_name[:-4]

    # np_array = np.transpose(np_array, (2, 1, 0))
    # np_array[np_array < 0.5] = 0
    # np_array[np_array >= 0.5] = 1
    print(np_array.shape)
    np_array = np_array.astype("uint8")
    header = io.Header(spacing=spacing)
    print("mha file path:", save_dict + stl_name + ".mha")
    io.save(np_array, save_dict + stl_name + ".mha", hdr=header, use_compression=True)
    print("saved")
    stl_path = save_dict + stl_name + ".stl"
    convert_mha_to_stl(save_dict + stl_name + ".mha", stl_path, visualize=visualize)


def visualize_numpy_as_stl(
    numpy_array, temp_path="/home/zhoul0a/Desktop/transfer/temp.stl", spacing=(1, 1, 1)
):
    # temp_path: we need to save numpy_array as .stl, then load .stl to visualize
    # numpy_array should be binary, like 1 means inside tracheae, 0 means outside tracheae
    save_dict = temp_path[: -len(temp_path.split("/")[-1])]
    stl_name = temp_path.split("/")[-1]
    save_numpy_as_stl(numpy_array, save_dict, stl_name, True, spacing=spacing)


def visualize_enhanced_channels(array_with_enhanced_channel_dict, save_dict):
    arrays_name_list = os.listdir(array_with_enhanced_channel_dict)
    for array_name in arrays_name_list:
        array_with_enhanced_channel = np.load(
            os.path.join(array_with_enhanced_channel_dict, array_name)
        )["array"]
        high_recall_mask = array_with_enhanced_channel[:, :, :, 1]
        save_numpy_as_stl(
            high_recall_mask,
            os.path.join(save_dict, "high_recall/"),
            array_name[:-4] + "_high_recall.stl",
        )

        high_precision_mask = array_with_enhanced_channel[:, :, :, 2]
        save_numpy_as_stl(
            high_precision_mask,
            os.path.join(save_dict, "high_precision/"),
            array_name[:-4] + "_high_precision.stl",
        )


if __name__ == "__main__":

    # array = Functions.read_in_mha(
    #     "/home/zhoul0a/Desktop/wrong_patients/ps000007/2020-06-11/Data/ground_truth/vein.mha"
    # )
    # visualize_numpy_as_stl(array)
    # print(np.shape(array))

    exit()
    # visualize_numpy_as_stl(array)
    # visualize_stl("/home/zhoul0a/Desktop/transfer/temp.stl")
    # exit()
    visualize_stl(
        "/home/zhoul0a/Desktop/Lung_CAD_NMI/applications/tracheae_seg/visualization_threshold2_7connected/Normal/A12/tracheae_segmentation.stl"
    )
    visualize_stl(
        "/home/zhoul0a/Desktop/Lung_CAD_NMI/applications/tracheae_seg/visualization_percentile98.5_10connected/Normal/A12/tracheae_segmentation.stl"
    )
    visualize_stl(
        "/home/zhoul0a/Desktop/Lung_CAD_NMI/applications/tracheae_seg/visualization_v1/Normal/A12/tracheae_segmentation.stl"
    )
    visualize_stl(
        "/home/zhoul0a/Desktop/Lung_CAD_NMI/applications/tracheae_seg/visualization_threshold1.85_10connected/A12/tracheae_segmentation.stl"
    )

    exit()
    array = np.load(
        "/home/zhoul0a/Desktop/Lung_CAD_NMI/raw_data/blood_vessel/arrays_raw/xwqg-A000032_2019-09-29.npy"
    )
    array = array[:, :, :, 1]
    visualize_numpy_as_stl(array)
    exit()
