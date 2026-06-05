"""
데이터 전처리 모듈
data/preprocessor.py에 저장
"""

import numpy as np
from typing import Dict, List
import logging


class DataPreprocessor:
    """데이터 전처리기"""

    def separate_by_component(self, dataset: List[Dict]) -> Dict[str, List[Dict]]:
        """성분별로 데이터 분리"""

        separated = {
            'bond': [],
            'angle': [],
            'torsion': [],
            'vdw': []
        }

        for sample in dataset:
            # scan_type 필드가 있으면 사용
            if 'scan_type' in sample:
                scan_type = sample['scan_type']
                if scan_type in separated:
                    # 성분별 에너지 추정
                    self._estimate_component_energies(sample)
                    separated[scan_type].append(sample)
            else:
                # scan_type이 없으면 구조로 추정
                component = self._infer_component(sample)
                if component:
                    self._estimate_component_energies(sample)
                    separated[component].append(sample)

        return separated

    def _infer_component(self, sample: Dict) -> str:
        """구조로부터 성분 추정"""

        n_atoms = len(sample['types'])
        n_pairs = len(sample.get('pairs', []))
        n_triplets = len(sample.get('triplets', []))
        n_quads = len(sample.get('quads', []))

        # 2원자 시스템은 bond
        if n_atoms == 2:
            return 'bond'

        # 3원자 시스템
        elif n_atoms == 3:
            # 거리 확인으로 bond vs angle 구분
            R = sample['R']
            dist1 = np.linalg.norm(R[1] - R[0])
            dist2 = np.linalg.norm(R[2] - R[1])

            if dist1 < 3.0 and dist2 < 3.0:
                return 'angle'
            else:
                return 'vdw'

        # 4원자 이상
        elif n_atoms >= 4:
            if n_quads > 0:
                return 'torsion'
            else:
                return 'vdw'

        return 'vdw'

    def _estimate_component_energies(self, sample: Dict):
        """성분별 에너지 추정"""

        E_total = sample['E_dft']
        scan_type = sample.get('scan_type', 'unknown')

        # 성분별 가중치
        if scan_type == 'bond':
            sample['E_bond'] = E_total * 0.7
            sample['E_angle'] = E_total * 0.2
            sample['E_vdw'] = E_total * 0.1
            sample['E_torsion'] = 0.0

        elif scan_type == 'angle':
            sample['E_bond'] = E_total * 0.5
            sample['E_angle'] = E_total * 0.35
            sample['E_vdw'] = E_total * 0.15
            sample['E_torsion'] = 0.0

        elif scan_type == 'torsion':
            sample['E_bond'] = E_total * 0.4
            sample['E_angle'] = E_total * 0.25
            sample['E_torsion'] = E_total * 0.25
            sample['E_vdw'] = E_total * 0.1

        elif scan_type == 'vdw':
            sample['E_bond'] = 0.0
            sample['E_angle'] = 0.0
            sample['E_torsion'] = 0.0
            sample['E_vdw'] = E_total

        else:
            # 기본 분배
            sample['E_bond'] = E_total * 0.5
            sample['E_angle'] = E_total * 0.2
            sample['E_vdw'] = E_total * 0.2
            sample['E_torsion'] = E_total * 0.1


class DataValidator:
    """데이터 검증기"""

    def validate_and_clean(self, dataset: List[Dict]) -> List[Dict]:
        """데이터 검증 및 정리"""

        cleaned = []

        for i, sample in enumerate(dataset):
            # 에너지 검증
            E_dft = sample.get('E_dft', 0)

            if abs(E_dft) > 1000:
                logging.warning(f"샘플 {i}: 에너지 너무 큼 ({E_dft:.2f} eV)")
                continue

            if abs(E_dft) < 0.001:
                logging.warning(f"샘플 {i}: 에너지 너무 작음 ({E_dft:.2f} eV)")
                continue

            # 구조 검증
            R = sample.get('R')
            if R is None or len(R) == 0:
                logging.warning(f"샘플 {i}: 좌표 없음")
                continue

            if np.any(np.isnan(R)) or np.any(np.isinf(R)):
                logging.warning(f"샘플 {i}: 잘못된 좌표")
                continue

            # 연결성 검증/생성
            if 'pairs' not in sample or len(sample['pairs']) == 0:
                sample['pairs'] = self._generate_pairs(R)

            if 'triplets' not in sample:
                sample['triplets'] = self._generate_triplets(R, sample['pairs'])

            if 'quads' not in sample:
                sample['quads'] = np.zeros((0, 4), dtype=np.int32)

            cleaned.append(sample)

        logging.info(f"데이터 검증: {len(cleaned)}/{len(dataset)} 유효")

        return cleaned

    def _generate_pairs(self, R: np.ndarray, cutoff: float = 5.0) -> np.ndarray:
        """원자 쌍 생성"""

        n_atoms = len(R)
        pairs = []

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(R[i] - R[j])
                if dist < cutoff:
                    pairs.append([i, j])

        return np.array(pairs, dtype=np.int32) if pairs else np.zeros((0, 2), dtype=np.int32)

    def _generate_triplets(self, R: np.ndarray, pairs: np.ndarray) -> np.ndarray:
        """3원자 조합 생성"""

        triplets = []
        n_atoms = len(R)

        # 각 원자를 중심으로
        for j in range(n_atoms):
            neighbors = []

            # j와 연결된 원자 찾기
            for pair in pairs:
                if pair[0] == j:
                    neighbors.append(pair[1])
                elif pair[1] == j:
                    neighbors.append(pair[0])

            # 이웃 원자들로 삼중항 생성
            for idx_i, i in enumerate(neighbors):
                for k in neighbors[idx_i + 1:]:
                    triplets.append([i, j, k])

        return np.array(triplets, dtype=np.int32) if triplets else np.zeros((0, 3), dtype=np.int32)