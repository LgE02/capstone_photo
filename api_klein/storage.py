"""S3 어댑터 — 동화별 폴더로 레퍼런스/페이지 이미지·메타데이터 저장.

S3 폴더 구조:
    fairytales/{fairytaleId}/
      bible.json                       (캐릭터 visual_description + setting + character_type)
      references/HERO.png
      references/VILLAIN.png
      references/DISPATCHER.png
      pages/page_NN.png
"""

from __future__ import annotations

import io
import json
import os
from typing import Any

import boto3
from botocore.exceptions import ClientError
from PIL import Image


# Propp 민담 형태론 표준 역할들. 메시지에 어느 부분집합이든 들어올 수 있음.
KNOWN_ROLES = ("HERO", "VILLAIN", "HELPER", "DISPATCHER", "FALSE_HERO", "DONOR")

# 호환을 위한 alias (구 API 사용처). 새 코드는 KNOWN_ROLES 또는 message['characters'].keys() 사용 권장.
ROLES = KNOWN_ROLES


class S3Storage:
    """S3 버킷 한 개에 동화별 폴더로 결과물을 저장.

    환경변수 비어 있어도 객체 생성은 됨 (boto3가 실제 호출 시 인증 실패).
    개발 단계에서 키 없이 import만 가능.
    """

    def __init__(
        self,
        bucket: str | None = None,
        region: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
    ):
        self.bucket = bucket or os.environ.get("S3_BUCKET", "")
        self.region = region or os.environ.get("AWS_REGION", "ap-northeast-2")

        client_kwargs: dict[str, Any] = {"region_name": self.region}
        ak = access_key or os.environ.get("AWS_ACCESS_KEY_ID")
        sk = secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
        if ak and sk:
            client_kwargs["aws_access_key_id"] = ak
            client_kwargs["aws_secret_access_key"] = sk

        self.client = boto3.client("s3", **client_kwargs)

    # ── 경로 헬퍼 ──────────────────────────────────────────────────────────

    @staticmethod
    def _fairytale_prefix(fairytale_id: int) -> str:
        return f"fairytales/{fairytale_id}"

    @classmethod
    def reference_key(cls, fairytale_id: int, role: str) -> str:
        return f"{cls._fairytale_prefix(fairytale_id)}/references/{role}.png"

    @classmethod
    def page_key(cls, fairytale_id: int, page_no: int) -> str:
        return f"{cls._fairytale_prefix(fairytale_id)}/pages/page_{page_no:02d}.png"

    @classmethod
    def bible_key(cls, fairytale_id: int) -> str:
        return f"{cls._fairytale_prefix(fairytale_id)}/bible.json"

    def s3_url(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"

    def public_url(self, key: str) -> str:
        """공개 버킷일 때 사용. 비공개면 보통 presigned URL 또는 s3:// 형태로 전달."""
        return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{key}"

    # ── 이미지 ─────────────────────────────────────────────────────────────

    def upload_image(self, key: str, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buf.getvalue(),
            ContentType="image/png",
        )
        return self.s3_url(key)

    def download_image(self, key: str) -> Image.Image | None:
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise
        return Image.open(io.BytesIO(obj["Body"].read())).convert("RGB")

    def upload_reference(self, fairytale_id: int, role: str, image: Image.Image) -> str:
        return self.upload_image(self.reference_key(fairytale_id, role), image)

    def download_reference(self, fairytale_id: int, role: str) -> Image.Image | None:
        return self.download_image(self.reference_key(fairytale_id, role))

    def upload_page(self, fairytale_id: int, page_no: int, image: Image.Image) -> str:
        """페이지 이미지 업로드 → 프론트가 그대로 <img src> 에 쓸 수 있는 https URL 반환.

        ※ 버킷이 public-read 가 아니면 이 URL 로 직접 접근 안 됨. 둘 중 택일:
          1) 버킷 정책에 public-read 적용 (가장 단순, 데모용)
          2) presigned_page_url() 로 만료 시간 있는 서명 URL 발급 (운영 권장)
        """
        self.upload_image(self.page_key(fairytale_id, page_no), image)
        return self.public_url(self.page_key(fairytale_id, page_no))

    def page_exists(self, fairytale_id: int, page_no: int) -> bool:
        return self.object_exists(self.page_key(fairytale_id, page_no))

    def page_url(self, fairytale_id: int, page_no: int) -> str:
        """이미 업로드된 페이지의 https URL (멱등 처리용 — 재생성 없이 기존 URL 반환)."""
        return self.public_url(self.page_key(fairytale_id, page_no))

    def presigned_page_url(self, fairytale_id: int, page_no: int, expires_in: int = 3600) -> str:
        """비공개 버킷일 때 사용 — 만료 있는 서명 URL.

        Spring 측에서 'https URL 받아도 403 뜬다' 하면 worker.py 가 publish 하기
        전에 page_url 대신 이걸 부르도록 바꾸면 됨.
        """
        return self.client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": self.bucket, "Key": self.page_key(fairytale_id, page_no)},
            ExpiresIn=expires_in,
        )

    # ── JSON (bible) ───────────────────────────────────────────────────────

    def upload_bible(self, fairytale_id: int, data: dict[str, Any]) -> str:
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.client.put_object(
            Bucket=self.bucket,
            Key=self.bible_key(fairytale_id),
            Body=body,
            ContentType="application/json; charset=utf-8",
        )
        return self.s3_url(self.bible_key(fairytale_id))

    def download_bible(self, fairytale_id: int) -> dict[str, Any] | None:
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=self.bible_key(fairytale_id))
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise
        return json.loads(obj["Body"].read())

    # ── 존재 여부 체크 ─────────────────────────────────────────────────────

    def object_exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound"):
                return False
            raise

    def references_exist(self, fairytale_id: int, roles: list[str]) -> bool:
        """주어진 역할 리스트 전부 S3에 존재하는가."""
        if not roles:
            return False
        return all(
            self.object_exists(self.reference_key(fairytale_id, role))
            for role in roles
        )
