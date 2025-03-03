import mmcv
import mmengine.fileio as fileio
import tifffile
import numpy as np
import io
import cv2
from .pfm_io import pfm_from_bytes, bytes_from_pfm


def disp_from_byte(data_bytes, imdecode_backend='pfm'):
    """

    Args:
        data_bytes:
        imdecode_backend:

    Returns:

    """
    if imdecode_backend == 'pfm':
        raw_disp = pfm_from_bytes(data_bytes)
        if raw_disp.ndim == 3:  # [h,w,c]
            raw_disp = raw_disp[..., 0]  # if flow map: raw_disp = raw_disp[..., :2]
    else:
        raw_disp = mmcv.imfrombytes(
            data_bytes,
            flag='unchanged',
            backend=imdecode_backend
        ).squeeze()
    return raw_disp


def disp_read(file_path, *, backend_args=None, imdecode_backend='pfm'):
    ext = file_path.split('.')[-1].lower()
    if ext == 'pfm':
        assert imdecode_backend == 'pfm', "'pfm' format file only support 'pfm' imencode_backend"
    elif ext in ['tif', 'tiff']:
        assert imdecode_backend in ['cv2', 'tiff'], \
            f"'{ext}' format file only support 'cv2' or 'tiff' imencode_backend"
    else:
        assert imdecode_backend == 'cv2', \
            f"'{ext}' format file only support 'cv2' imencode_backend"
    disp_bytes = fileio.get(file_path, backend_args)
    return disp_from_byte(disp_bytes, imdecode_backend)


def disp_to_byte(disp_map, imencode_backend='pfm'):
    """Convert disparity map to bytes.

    Args:
        disp_map (ndarray): Input disparity map
        imencode_backend (str): The encoding backend, supports:
            - 'pfm': PFM format
            - 'cv2': OpenCV backend
            - 'tiff': TIFF format using tifffile
            Defaults to 'pfm'.

    Returns:
        bytes: Encoded bytes of the disparity map.

    Raises:
        ValueError: If imencode_backend is not supported.
    """
    if imencode_backend == 'pfm':
        if disp_map.ndim == 2:  # [h,w]
            disp_map = disp_map[..., None]  # convert to [h,w,1]
        disp_bytes = bytes_from_pfm(disp_map.astype('float32'))
    elif imencode_backend == 'cv2':
        success, encoded_img = cv2.imencode('.png', disp_map)
        if not success:
            raise RuntimeError("Failed to encode image using cv2")
        disp_bytes = encoded_img.tobytes()
    elif imencode_backend == 'tiff':
        # Save to bytes using BytesIO
        bio = io.BytesIO()
        tifffile.imwrite(
            bio,
            disp_map.astype(np.float32),
            compression='zlib',  # zlib
            photometric='minisblack'  #
        )
        disp_bytes = bio.getvalue()
        bio.close()
    else:
        raise ValueError(f"Unsupported imencode_backend: {imencode_backend}, "
                         f"should be one of ['pfm', 'cv2', 'tiff']")
    return disp_bytes


def disp_write(disp_map, file_path, *, backend_args=None, imencode_backend='pfm'):
    """Write disparity map to file.

    Args:
        disp_map (ndarray): Disparity map to be written
        file_path (str): Destination file path
        backend_args (dict, optional): Backend arguments for mmengine.fileio
        imencode_backend (str): The encoding backend, supports:
            - 'pfm': PFM format
            - 'cv2': OpenCV backend
            - 'tiff': TIFF format using tifffile
            Defaults to 'pfm'.
    """
    ext = file_path.split('.')[-1].lower()

    #
    if ext == 'pfm':
        assert imencode_backend == 'pfm', "'pfm' format file only support 'pfm' imencode_backend"
    elif ext in ['tif', 'tiff']:
        assert imencode_backend in ['cv2', 'tiff'], \
            f"'{ext}' format file only support 'cv2' or 'tiff' imencode_backend"
    else:
        assert imencode_backend == 'cv2', \
            f"'{ext}' format file only support 'cv2' imencode_backend"

    disp_bytes = disp_to_byte(disp_map, imencode_backend)
    fileio.put(
        disp_bytes,
        file_path,
        backend_args=backend_args
    )
