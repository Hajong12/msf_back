# face_color_extraction_refactored.py
# 얼굴 부위별 색상 추출 - Segformer 기반, 다중 색상 추출, 시각화 기능 완전판

import numpy as np
import cv2
from PIL import Image
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from fastapi import UploadFile, HTTPException
import io
import os
from typing import List, Dict, Any, Optional

class FaceColorExtractor:
    def __init__(self):
        """얼굴 파싱 모델 초기화"""
        self.processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        self.model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
        
        self.label_map = {
            0: "background", 1: "skin", 2: "nose", 3: "eye_g", 4: "left_eye",
            5: "right_eye", 6: "left_brow", 7: "right_brow", 8: "left_ear", 9: "right_ear",
            10: "mouth", 11: "upper_lip", 12: "lower_lip", 13: "hair", 14: "hat",
            15: "earr_l", 16: "earr_r", 17: "neck_l", 18: "neck"
        }
        
        self.target_parts = {
            "eyes": [4, 5], "nose": [2], "lips": [10, 11, 12],
            "hair": [13], "skin": [1]
        }

    def validate_and_count_faces(self, segmentation_mask, original_image_shape):
        """세그멘테이션 마스크를 사용하여 유효한 단일 얼굴이 있는지 검증합니다."""
        unique_labels = np.unique(segmentation_mask)
        has_skin = 1 in unique_labels
        has_eye = 4 in unique_labels or 5 in unique_labels

        if not (has_skin and has_eye):
            raise HTTPException(status_code=400, detail="얼굴의 핵심 부위(피부, 눈)가 인식되지 않았습니다. 더 선명한 사진을 사용해주세요.")

        skin_mask = (segmentation_mask == 1).astype(np.uint8)
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_face_area = (original_image_shape[0] * original_image_shape[1]) * 0.01
        face_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_face_area]

        if len(face_contours) == 0:
            raise HTTPException(status_code=400, detail="얼굴을 찾을 수 없습니다. 조명이 밝고 얼굴이 잘 보이는 사진을 사용해주세요.")
        if len(face_contours) > 1:
            raise HTTPException(status_code=400, detail=f"{len(face_contours)}명의 얼굴이 감지되었습니다. 한 명의 얼굴만 있는 사진을 사용해주세요.")

        face_area = cv2.contourArea(face_contours[0])
        image_area = original_image_shape[0] * original_image_shape[1]
        face_ratio = face_area / image_area
        if face_ratio < 0.03:
            raise HTTPException(status_code=400, detail="얼굴이 너무 작습니다. 더 가까이서 찍은 사진을 사용해주세요.")
        print(f"얼굴 검증 완료: 1개의 얼굴 감지, 얼굴 비율: {face_ratio:.2%}")

    def parse_face_from_memory(self, image):
        """메모리의 이미지에서 얼굴을 파싱하고 검증합니다."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        predicted_segmentation = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0].numpy()
        self.validate_and_count_faces(predicted_segmentation, np.array(image).shape)
        return np.array(image), predicted_segmentation

    def extract_dominant_colors(self, image_region, n_colors=3) -> Optional[List[np.ndarray]]:
        """이미지 영역에서 가장 넓은 면적을 차지하는 주요 색상(팔레트)을 추출합니다."""
        if len(image_region) < n_colors: return None
        pixels = image_region.reshape(-1, 3)
        valid_pixels = pixels[(np.sum(pixels, axis=1) > 30) & (np.sum(pixels, axis=1) < 720)]
        if len(valid_pixels) < n_colors: return None
        
        unique_colors = np.unique(valid_pixels, axis=0)
        if len(unique_colors) < n_colors:
            n_colors = len(unique_colors)

        # K-Means를 실행하여 5개의 후보군을 찾음 (선택의 폭을 넓히기 위해)
        kmeans = KMeans(n_clusters=min(n_colors + 2, len(unique_colors)), random_state=42, n_init='auto')
        kmeans.fit(valid_pixels)
        
        # 각 클러스터(색상)가 차지하는 픽셀 수를 계산
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        
        # 픽셀 수가 많은 순서대로 클러스터의 인덱스를 정렬
        sorted_indices = [i for _, i in sorted(zip(counts, unique), reverse=True)]
        
        # 가장 면적이 넓은 상위 n_colors개의 색상을 선택
        dominant_centers = kmeans.cluster_centers_[sorted_indices[:n_colors]]
        
        return dominant_centers.astype(int)

    def extract_pupil_color(self, image_region) -> Optional[np.ndarray]:
        """눈 영역에서 동공 색상(가장 어두운 색) 하나만 추출합니다."""
        if len(image_region) == 0: return None
        pixels = image_region.reshape(-1, 3)
        brightness = np.mean(pixels, axis=1)
        dark_threshold = np.percentile(brightness, 20)
        dark_pixels = pixels[brightness < dark_threshold]
        if len(dark_pixels) == 0: 
            colors = self.extract_dominant_colors(image_region, n_colors=1)
            return colors[0] if colors else None
        return np.mean(dark_pixels, axis=0).astype(int)

    def rgb_to_hex(self, rgb: np.ndarray) -> str:
        """RGB를 HEX 코드로 변환"""
        return f"{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    def extract_face_colors(self, image: Image.Image) -> Dict[str, Any]:
        """얼굴 부위별 색상 팔레트를 추출하는 메인 함수"""
        original_image, segmentation_mask = self.parse_face_from_memory(image)
        
        results = {}
        for part_name, label_ids in self.target_parts.items():
            part_mask = np.isin(segmentation_mask, label_ids)
            if np.any(part_mask):
                part_pixels = original_image[part_mask]
                if part_name == "eyes":
                    dominant_color = self.extract_pupil_color(part_pixels)
                    colors_list = [dominant_color] if dominant_color is not None else []
                else:
                    colors_list = self.extract_dominant_colors(part_pixels, n_colors=3)
                
                results[part_name] = {
                    "rgb": [c.tolist() for c in colors_list] if colors_list is not None else [],
                    "hex": [self.rgb_to_hex(c) for c in colors_list] if colors_list is not None else []
                }
            else:
                results[part_name] = {"rgb": [], "hex": []}
        
        return results, original_image, segmentation_mask

    def _plot_color_palette(self, axes, colors):
        """색상 팔레트 시각화 (다중 색상 지원)"""
        axes.axis('off')
        axes.set_title("Extracted Color Palettes")
        y_pos = 0.95
        for part_name, color_info in colors.items():
            hex_codes = color_info.get("hex", [])
            if not hex_codes:
                continue
            
            axes.text(0.05, y_pos, f"{part_name.title()}:", fontsize=10, va='center', weight='bold')
            
            for i, hex_code in enumerate(hex_codes):
                rgb_color = color_info["rgb"][i]
                color_rgb_normalized = [c/255 for c in rgb_color]
                # xy 좌표를 튜플 ((x, y)) 형태로 전달해야 합니다.
                axes.add_patch(plt.Rectangle(((0.3 + i * 0.2), y_pos - 0.04), 0.15, 0.08, 
                                          facecolor=color_rgb_normalized, edgecolor='black'))
                axes.text(0.375 + i * 0.2, y_pos - 0.07, f"#{hex_code}", fontsize=8, ha='center')
            
            y_pos -= 0.2
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)

    def _plot_part_masks(self, axes, original_image, segmentation_mask):
        """부위별 마스크 시각화"""
        part_names = list(self.target_parts.keys())
        for i, part_name in enumerate(part_names[:3]):
            ax = axes[1, i]
            label_ids = self.target_parts[part_name]
            part_mask = np.isin(segmentation_mask, label_ids)
            
            ax.imshow(original_image)
            if np.any(part_mask):
                colored_mask = np.zeros_like(original_image)
                colored_mask[part_mask] = [255, 0, 0] # Red
                ax.imshow(colored_mask, alpha=0.5)
            
            ax.set_title(f"{part_name.title()} Mask")
            ax.axis('off')

    def visualize_results(self, original_image, segmentation_mask, colors, save_path=None):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(segmentation_mask, cmap='tab20')
        axes[0, 1].set_title("Segmentation Mask")
        axes[0, 1].axis('off')
        
        self._plot_color_palette(axes[0, 2], colors)
        self._plot_part_masks(axes, original_image, segmentation_mask)

        axes[1, 2].axis('off')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"분석 결과 이미지를 '{save_path}'에 저장했습니다.")

async def main(file: UploadFile):
    """얼굴 이미지에서 색상 팔레트를 추출하고 시각화하는 메인 함수"""
    extractor = FaceColorExtractor()
    face_colors = {part: [] for part in extractor.target_parts.keys()}
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        colors_data, original_image, segmentation_mask = extractor.extract_face_colors(image)
        
        for part_name, color_info in colors_data.items():
            if color_info["hex"]:
                face_colors[part_name] = color_info["hex"]
        
        # 시각화 함수 호출 활성화
        extractor.visualize_results(original_image, segmentation_mask, colors_data, "face_color_analysis.png")
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"이미지 처리 중 오류가 발생했습니다: {str(e)}")
    
    from schemas.personal_schema import FaceColor
    return FaceColor(**face_colors)

async def extract_face_only(file: UploadFile):
    """얼굴만 추출하고 배경을 제거하는 함수"""
    extractor = FaceColorExtractor()
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        original_image, segmentation_mask = extractor.parse_face_from_memory(image)
        
        face_labels = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        face_mask = np.isin(segmentation_mask, face_labels)
        
        if not np.any(face_mask):
            raise HTTPException(status_code=400, detail="얼굴 영역을 찾을 수 없습니다.")
        
        face_only_image = np.zeros((original_image.shape[0], original_image.shape[1], 4), dtype=np.uint8)
        face_only_image[:, :, :3] = original_image
        face_only_image[:, :, 3] = face_mask.astype(np.uint8) * 255
        
        face_coords = np.where(face_mask)
        y_min, y_max = np.min(face_coords[0]), np.max(face_coords[0])
        x_min, x_max = np.min(face_coords[1]), np.max(face_coords[1])
        
        cropped_face = face_only_image[y_min:y_max+1, x_min:x_max+1]
        cropped_pil = Image.fromarray(cropped_face, 'RGBA')
        
        img_byte_arr = io.BytesIO()
        cropped_pil.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return img_byte_arr.getvalue()
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"얼굴 추출 중 예상치 못한 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"이미지 처리 중 오류가 발생했습니다: {str(e)}")