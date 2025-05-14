from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.similarity.pairwise.lightglue import MatchLightGlue
from wildlife_tools.similarity.wildfusion import SimilarityPipeline
from wildlife_tools.features import DeepFeatures
from wildlife_tools.features.local import AlikedExtractor
from wildlife_tools.similarity.calibration import IsotonicCalibration
from wildlife_tools.similarity.pairwise.loftr import MatchLOFTR
from wildlife_tools.features.local import DiskExtractor, SuperPointExtractor


'''
MegaDescriptor, ALIKED matcher 각각 생성 + return
'''
def build_megadescriptor(model, transform, device='cuda', batch_size=16):
    return SimilarityPipeline(
        matcher=CosineSimilarity(),
        extractor=DeepFeatures(model=model, device=device, batch_size=batch_size),
        transform=transform,
        calibration=IsotonicCalibration()
    )

def build_aliked(transform, device='cuda', batch_size=16):
    return SimilarityPipeline(
        matcher=MatchLightGlue(features='aliked', device=device, batch_size=batch_size),
        extractor=AlikedExtractor(),
        transform=transform,
        calibration=IsotonicCalibration()
    )
    
# LightGlue + DISK
def build_disk(transform, device='cuda', batch_size=16):
    return SimilarityPipeline(
        matcher=MatchLightGlue(features='disk', device=device, batch_size=batch_size),
        extractor=DiskExtractor(),
        transform=transform,
        calibration=IsotonicCalibration()
    )

# LightGlue + SuperPoint
def build_superpoint(transform, device='cuda', batch_size=16):
    return SimilarityPipeline(
        matcher=MatchLightGlue(features='superpoint', device=device, batch_size=batch_size),
        extractor=SuperPointExtractor(),
        transform=transform,
        calibration=IsotonicCalibration()
    )

# LoFTR
def build_loftr(transform, device='cuda', batch_size=16):
    return SimilarityPipeline(
        matcher=MatchLoFTR(model_type='outdoor', device=device),
        extractor=None,  # LoFTR 자체적으로 추출+매칭
        transform=transform,
        calibration=IsotonicCalibration()
    )
