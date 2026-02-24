#!/usr/bin/env python3
"""
Stable Diffusion Image Client
=============================
A local Python client that sends prompts to a remote Stable Diffusion API
running on Google Colab and saves the generated images.

Usage:
    python image_client.py "cyberpunk city at night"
    python image_client.py "a cat sitting on a sofa" --width 768 --height 768
    python image_client.py "sunset over mountains" --steps 50 --seed 42

Requirements:
    pip install -r requirements.txt
"""

import argparse
import base64
import io
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import requests
from PIL import Image


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_API_URL = None  # Set via CLI or environment variable
DEFAULT_TIMEOUT = 180  # seconds (generation can take time)
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_STEPS = 25
DEFAULT_GUIDANCE = 7.5
OUTPUT_DIR = "generated_images"


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class ImageGenerationError(Exception):
    """Base exception for image generation errors"""
    pass


class APIConnectionError(ImageGenerationError):
    """Failed to connect to the API"""
    pass


class APITimeoutError(ImageGenerationError):
    """API request timed out"""
    pass


class InvalidResponseError(ImageGenerationError):
    """Invalid response from API"""
    pass


# ============================================================================
# CLIENT CLASS
# ============================================================================

class StableDiffusionClient:
    """
    Client for interacting with the remote Stable Diffusion API.
    
    Handles connection, request formatting, response parsing,
    and error handling.
    """
    
    def __init__(
        self,
        api_url: str,
        timeout: int = DEFAULT_TIMEOUT,
        output_dir: str = OUTPUT_DIR
    ):
        """
        Initialize the client.
        
        Args:
            api_url: The base URL of the API (ngrok URL)
            timeout: Request timeout in seconds
            output_dir: Directory to save generated images
        """
        if not api_url:
            raise ValueError("API URL is required")
        
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.output_dir = Path(output_dir)
        self._session = requests.Session()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _validate_prompt(self, prompt: str) -> None:
        """
        Validate the input prompt.
        
        Args:
            prompt: The text prompt to validate
            
        Raises:
            ValueError: If prompt is invalid
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        
        if not prompt.strip():
            raise ValueError("Prompt cannot be only whitespace")
        
        if len(prompt) > 500:
            raise ValueError("Prompt cannot exceed 500 characters")
    
    def _validate_params(self, **params) -> None:
        """
        Validate generation parameters.
        
        Args:
            **params: Generation parameters to validate
            
        Raises:
            ValueError: If any parameter is invalid
        """
        if "height" in params:
            h = params["height"]
            if not (256 <= h <= 1024):
                raise ValueError("Height must be between 256 and 1024")
        
        if "width" in params:
            w = params["width"]
            if not (256 <= w <= 1024):
                raise ValueError("Width must be between 256 and 1024")
        
        if "num_inference_steps" in params:
            steps = params["num_inference_steps"]
            if not (1 <= steps <= 100):
                raise ValueError("Steps must be between 1 and 100")
        
        if "guidance_scale" in params:
            guidance = params["guidance_scale"]
            if not (1.0 <= guidance <= 20.0):
                raise ValueError("Guidance scale must be between 1.0 and 20.0")
    
    def _check_connection(self) -> bool:
        """
        Check if the API is accessible.
        
        Returns:
            True if connected, False otherwise
        """
        try:
            response = self._session.get(
                f"{self.api_url}/health",
                timeout=10
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        steps: int = DEFAULT_STEPS,
        guidance: float = DEFAULT_GUIDANCE,
        seed: Optional[int] = None,
        save: bool = True
    ) -> Dict[str, Any]:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text description of the desired image
            negative_prompt: Things to avoid in the image
            width: Image width (256-1024)
            height: Image height (256-1024)
            steps: Number of inference steps (1-100)
            guidance: Guidance scale (1.0-20.0)
            seed: Random seed for reproducibility
            save: Whether to save the image locally
            
        Returns:
            Dictionary containing:
                - image: PIL Image object
                - seed: Used seed
                - inference_time: Time taken in seconds
                - image_path: Path to saved image (if save=True)
                
        Raises:
            APIConnectionError: If cannot connect to API
            APITimeoutError: If request times out
            InvalidResponseError: If API returns invalid response
            ImageGenerationError: If generation fails
        """
        # Validate inputs
        self._validate_prompt(prompt)
        self._validate_params(
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance
        )
        
        # Check connection first
        print("🔌 Checking API connection...")
        if not self._check_connection():
            raise APIConnectionError(
                f"Cannot connect to API at {self.api_url}. "
                "Make sure the Colab server is running."
            )
        print("✅ Connected to API")
        
        # Prepare request
        print(f"🎨 Generating: '{prompt}'")
        
        data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
        }
        
        if seed is not None:
            data["seed"] = seed
        
        # Make request
        start_time = time.time()
        
        try:
            response = self._session.post(
                f"{self.api_url}/generate",
                data=data,
                timeout=self.timeout
            )
        except requests.Timeout as e:
            raise APITimeoutError(
                f"Request timed out after {self.timeout}s"
            ) from e
        except requests.ConnectionError as e:
            raise APIConnectionError(
                f"Failed to connect to API: {e}"
            ) from e
        except requests.RequestException as e:
            raise ImageGenerationError(
                f"Request failed: {e}"
            ) from e
        
        # Check response status
        if response.status_code != 200:
            try:
                error_detail = response.json().get("detail", response.text)
            except ValueError:
                error_detail = response.text
            raise ImageGenerationError(f"API error: {error_detail}")
        
        # Parse response
        try:
            result = response.json()
        except ValueError as e:
            raise InvalidResponseError(
                f"Invalid JSON response: {e}"
            ) from e
        
        if not result.get("success"):
            raise ImageGenerationError(
                result.get("message", "Generation failed")
            )
        
        # Decode image
        try:
            image_base64 = result["image_base64"]
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise InvalidResponseError(
                f"Failed to decode image: {e}"
            ) from e
        
        inference_time = result.get("inference_time", time.time() - start_time)
        used_seed = result.get("seed", seed)
        
        print(f"✅ Generated in {inference_time:.2f}s (seed: {used_seed})")
        
        # Save image
        image_path = None
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = self.output_dir / f"output_{timestamp}.png"
            image.save(image_path)
            print(f"💾 Saved: {image_path}")
        
        return {
            "image": image,
            "seed": used_seed,
            "inference_time": inference_time,
            "image_path": image_path
        }


# ============================================================================
# CLI INTERFACE
# ============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate images using Stable Diffusion API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s "a cat sitting on a sofa"
    %(prog)s "cyberpunk city at night" --width 768 --height 768
    %(prog)s "sunset" --steps 50 --seed 42 --api-url https://your-ngrok-url.ngrok.io

Environment Variables:
    SD_API_URL    Override default API URL
        """
    )
    
    parser.add_argument(
        "prompt",
        nargs="?",
        type=str,
        help="Text prompt describing the image to generate"
    )
    
    parser.add_argument(
        "--api-url", "-u",
        type=str,
        default=os.environ.get("SD_API_URL"),
        help="API URL (ngrok URL). Can also set SD_API_URL env var"
    )
    
    parser.add_argument(
        "--width", "-w",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Image width (256-1024, default: {DEFAULT_WIDTH})"
    )
    
    parser.add_argument(
        "--height", "-h",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Image height (256-1024, default: {DEFAULT_HEIGHT})"
    )
    
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Number of inference steps (1-100, default: {DEFAULT_STEPS})"
    )
    
    parser.add_argument(
        "--guidance", "-g",
        type=float,
        default=DEFAULT_GUIDANCE,
        help=f"Guidance scale (1.0-20.0, default: {DEFAULT_GUIDANCE})"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--negative", "-n",
        type=str,
        default=None,
        help="Negative prompt (things to avoid)"
    )
    
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save image locally"
    )
    
    return parser.parse_args()


def main() -> int:
    """
    Main entry point.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_args()
    
    # Check for prompt
    if not args.prompt:
        print("Error: Prompt is required", file=sys.stderr)
        print("Usage: python image_client.py \"your prompt here\"", file=sys.stderr)
        print("Run with --help for more information", file=sys.stderr)
        return 1
    
    # Check for API URL
    if not args.api_url:
        print("Error: API URL is required", file=sys.stderr)
        print("Provide it via --api-url or SD_API_URL environment variable", file=sys.stderr)
        print("Example: python image_client.py \"prompt\" --api-url https://abc123.ngrok.io", file=sys.stderr)
        return 1
    
    # Create client
    try:
        client = StableDiffusionClient(
            api_url=args.api_url,
            timeout=args.timeout,
            output_dir=args.output_dir
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Generate image
    try:
        result = client.generate(
            prompt=args.prompt,
            negative_prompt=args.negative,
            width=args.width,
            height=args.height,
            steps=args.steps,
            guidance=args.guidance,
            seed=args.seed,
            save=not args.no_save
        )
        
        print("\n" + "="*50)
        print("✨ SUCCESS")
        print("="*50)
        print(f"Prompt:     {args.prompt}")
        print(f"Seed:       {result['seed']}")
        print(f"Time:       {result['inference_time']:.2f}s")
        if result['image_path']:
            print(f"Saved to:   {result['image_path']}")
        print("="*50)
        
        return 0
        
    except APIConnectionError as e:
        print(f"❌ Connection Error: {e}", file=sys.stderr)
        print("\nMake sure:", file=sys.stderr)
        print("  1. The Colab notebook is running", file=sys.stderr)
        print("  2. The ngrok URL is correct", file=sys.stderr)
        print("  3. Ngrok tunnel is active", file=sys.stderr)
        return 1
        
    except APITimeoutError as e:
        print(f"❌ Timeout: {e}", file=sys.stderr)
        print("Try increasing timeout with --timeout", file=sys.stderr)
        return 1
        
    except ImageGenerationError as e:
        print(f"❌ Generation Error: {e}", file=sys.stderr)
        return 1
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user", file=sys.stderr)
        return 1
        
    except Exception as e:
        print(f"❌ Unexpected Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
