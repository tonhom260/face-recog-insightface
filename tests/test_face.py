import io
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest

from attendance.config import Settings
from attendance.errors import APIError
from attendance.face import FaceService, create_sface_session


@pytest.fixture(scope="module")
def service():
    service=FaceService(Settings(inference_device="cpu"));service.load()
    assert service.ready, "Run scripts/prepare_models.py to obtain verified model assets"
    return service


def test_real_sface_parity_and_genuine_pair(service):
    embeddings=[]
    for path in ("face1.jpg","face2.jpg"):
        data=Path(path).read_bytes()
        embedding=service.extract(data)
        image=service.decode(data)
        height,width=image.shape[:2]
        if max(height,width)>1600:
            scale=1600/max(height,width)
            image=cv2.resize(image,(round(width*scale),round(height*scale)))
        service.detector.setInputSize((image.shape[1],image.shape[0]))
        faces=service.detector.detect(image)[1]
        aligned=service.aligner.alignCrop(image,faces[0].reshape(1,-1))
        reference=service.aligner.feature(aligned)
        assert service.similarity(embedding,reference)>0.99999
        embeddings.append(embedding)
    assert service.similarity(*embeddings)>service.settings.face_threshold
    assert service.device=="cpu" and service.providers==["CPUExecutionProvider"]


def test_actual_absent_multiple_faces(service):
    blank=np.zeros((600,600,3),dtype=np.uint8)
    data=cv2.imencode(".jpg",blank)[1].tobytes()
    with pytest.raises(APIError,match="No usable face") as error:
        service.extract(data)
    assert error.value.code=="FACE_NOT_FOUND"
    image=cv2.imread("face1.jpg")
    both=np.concatenate([image,image],axis=1)
    data=cv2.imencode(".jpg",both)[1].tobytes()
    with pytest.raises(APIError) as error:
        service.extract(data)
    assert error.value.code=="MULTIPLE_FACES"


@pytest.mark.parametrize("data",[b"",b"not-an-image",b"x"*(10*1024*1024+1)])
def test_invalid_mobile_image(service,data):
    with pytest.raises(APIError) as error:
        service.decode(data)
    assert error.value.code=="INVALID_IMAGE"


def test_exif_orientation_and_mobile_format(service):
    source=Image.new("RGB",(80,40),"red")
    exif=source.getexif();exif[274]=6
    output=io.BytesIO();source.save(output,format="JPEG",exif=exif)
    decoded=service.decode(output.getvalue())
    assert decoded.shape==(80,40,3)
    output=io.BytesIO();source.save(output,format="WEBP")
    assert service.decode(output.getvalue()).shape==(40,80,3)


class FakeRuntime:
    def __init__(self,providers,fail_cuda=False):
        self.available=providers;self.fail_cuda=fail_cuda;self.used=[]

    def get_available_providers(self):
        return self.available

    class SessionOptions:
        def add_session_config_entry(self,*args):
            pass

    def InferenceSession(self,path,sess_options,providers):
        names=[provider[0] if isinstance(provider,tuple) else provider for provider in providers]
        self.used.append(names[0])
        if providers==["CUDAExecutionProvider"] and self.fail_cuda:
            raise RuntimeError("GPU runtime failed")
        return SimpleNamespace(get_providers=lambda:names,get_inputs=lambda:[SimpleNamespace(name="input")],run=lambda *_:[np.ones((1,128))])


def test_device_cpu_does_not_choose_cuda():
    runtime=FakeRuntime(["CUDAExecutionProvider","CPUExecutionProvider"])
    _,device=create_sface_session("model","cpu",runtime)
    assert device=="cpu" and runtime.used==["CPUExecutionProvider"]


def test_device_explicit_cuda_unavailable():
    with pytest.raises(RuntimeError,match="unavailable"):
        create_sface_session("model","cuda",FakeRuntime(["CPUExecutionProvider"]))


def test_device_explicit_cuda_execution_failure():
    runtime=FakeRuntime(["CUDAExecutionProvider","CPUExecutionProvider"],True)
    with pytest.raises(RuntimeError,match="CUDA initialization/inference failed"):
        create_sface_session("model","cuda",runtime)
    assert runtime.used==["CUDAExecutionProvider"]


def test_device_auto_falls_back_on_cuda_failure():
    runtime=FakeRuntime(["CUDAExecutionProvider","CPUExecutionProvider"],True)
    _,device=create_sface_session("model","auto",runtime)
    assert device=="cpu" and runtime.used==["CUDAExecutionProvider","CPUExecutionProvider"]


def test_device_cuda_initialized_with_working_provider():
    runtime=FakeRuntime(["CUDAExecutionProvider","CPUExecutionProvider"])
    session,device=create_sface_session("model","cuda",runtime)
    assert device=="cuda" and session.get_providers()==["CUDAExecutionProvider"]


def test_missing_model_degrades_and_unavailable_endpoint(tmp_path):
    service=FaceService(Settings(model_dir=str(tmp_path),inference_device="cpu"));service.load()
    assert not service.ready
    with pytest.raises(APIError) as error:
        service.extract(b"image")
    assert error.value.status==503 and error.value.code=="MODELS_UNAVAILABLE"


def test_device_coreml_available_and_warmed():
    runtime=FakeRuntime(["CoreMLExecutionProvider","CPUExecutionProvider"])
    session,device=create_sface_session("model","coreml",runtime)
    assert device=="coreml"
    assert session.get_providers()==["CoreMLExecutionProvider","CPUExecutionProvider"]


def test_device_auto_prefers_coreml_before_cpu():
    runtime=FakeRuntime(["CoreMLExecutionProvider","CPUExecutionProvider"])
    _,device=create_sface_session("model","auto",runtime)
    assert device=="coreml" and runtime.used==["CoreMLExecutionProvider"]


def test_device_coreml_unavailable_fails_clearly():
    with pytest.raises(RuntimeError,match="CoreMLExecutionProvider is unavailable"):
        create_sface_session("model","coreml",FakeRuntime(["CPUExecutionProvider"]))


class FailingCoreMLRuntime(FakeRuntime):
    def InferenceSession(self,path,sess_options,providers):
        if isinstance(providers[0],tuple):
            self.used.append(providers[0][0])
            raise RuntimeError("CoreML model cannot initialize")
        return super().InferenceSession(path,sess_options,providers)


def test_coreml_initialization_failure_does_not_silently_use_cpu():
    runtime=FailingCoreMLRuntime(["CoreMLExecutionProvider","CPUExecutionProvider"])
    with pytest.raises(RuntimeError,match="COREML initialization/inference failed"):
        create_sface_session("model","coreml",runtime)
    assert runtime.used==["CoreMLExecutionProvider"]


def test_auto_coreml_failure_falls_back_to_cpu():
    runtime=FailingCoreMLRuntime(["CoreMLExecutionProvider","CPUExecutionProvider"])
    _,device=create_sface_session("model","auto",runtime)
    assert device=="cpu"
    assert runtime.used==["CoreMLExecutionProvider","CPUExecutionProvider"]
