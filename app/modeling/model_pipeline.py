"""
model_pipeline.py - Complete 3D Model Generation Pipeline for AI Smart Blackboard

This module orchestrates the entire 2D-to-3D conversion pipeline:
Shape Recognition → Geometry Generation → STL Export.

It acts as the central coordinator that connects the existing modules
without duplicating their implementation.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path for proper imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import Optional, Dict, Any, Union, List, Tuple
import time

# Import existing modules
from app.config import RECOGNITION_SETTINGS
from app.modeling.shape_recognizer import ShapeRecognizer, ShapeType, RecognitionResult
from app.modeling.geometry_generator import GeometryGenerator, GeometryResult
from app.modeling.stl_exporter import STLExporter


class ModelPipeline:
    """
    Complete 2D-to-3D conversion pipeline orchestrator.
    
    This class connects the ShapeRecognizer, GeometryGenerator, and STLExporter
    into a single streamlined pipeline. It handles validation, confidence
    checking, error handling, and result aggregation.
    
    The pipeline executes these steps:
    1. Input validation
    2. Shape recognition
    3. Confidence threshold check
    4. 3D geometry generation
    5. Mesh validation
    6. STL export
    7. Result verification
    
    All heavy lifting is delegated to the specialized modules.
    """
    
    def __init__(
        self,
        confidence_threshold: float = RECOGNITION_SETTINGS.shape_recognition_threshold,
        output_dir: Union[str, Path] = "app/data/models",
        pixels_per_mm: float = 5.0,
        default_height: float = 20.0
    ) -> None:
        """
        Initialize the ModelPipeline with configuration.
        
        Args:
            confidence_threshold: Minimum confidence for generation (0.0-1.0)
            output_dir: Directory for exported STL files
            pixels_per_mm: Pixel to millimeter conversion scale
            default_height: Default extrusion height for 3D models
        """
        self.confidence_threshold = confidence_threshold
        self.output_dir = Path(output_dir)
        self.pixels_per_mm = pixels_per_mm
        self.default_height = default_height
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self._logger = logging.getLogger(__name__)
        
        # Initialize components
        self._logger.info("Initializing ModelPipeline components...")
        self.shape_recognizer = ShapeRecognizer(debug=False)
        self.geometry_generator = GeometryGenerator()
        self.stl_exporter = STLExporter(output_dir=self.output_dir)
        
        # Configure geometry generator
        self.geometry_generator.set_scale(pixels_per_mm)
        self.geometry_generator.set_min_confidence(confidence_threshold)
        self.geometry_generator.config.default_height = default_height
        
        self._logger.info(
            f"Pipeline initialized with confidence threshold: {confidence_threshold}, "
            f"scale: {pixels_per_mm} px/mm, default height: {default_height} mm"
        )
        
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Update the confidence threshold.
        
        Args:
            threshold: New confidence threshold (0.0-1.0)
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        self.confidence_threshold = threshold
        self.geometry_generator.set_min_confidence(threshold)
        self._logger.info(f"Confidence threshold updated to: {threshold}")
        
    def set_pixels_per_mm(self, pixels_per_mm: float) -> None:
        """
        Update the pixel-to-millimeter scale.
        
        Args:
            pixels_per_mm: New scale value
        """
        if pixels_per_mm <= 0:
            raise ValueError("pixels_per_mm must be positive")
        self.pixels_per_mm = pixels_per_mm
        self.geometry_generator.set_scale(pixels_per_mm)
        self._logger.info(f"Pixel-to-mm scale updated to: {pixels_per_mm}")
        
    def set_default_height(self, height: float) -> None:
        """
        Update the default extrusion height.
        
        Args:
            height: New default height in mm
        """
        if height <= 0:
            raise ValueError("Default height must be positive")
        self.default_height = height
        self.geometry_generator.config.default_height = height
        self._logger.info(f"Default height updated to: {height} mm")
        
    def _validate_input(self, image_path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
        """
        Validate the input image path.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        if image_path is None:
            return False, "Image path is None"
            
        path = Path(image_path)
        
        if not path.exists():
            return False, f"Image file does not exist: {path}"
            
        if not path.is_file():
            return False, f"Path is not a file: {path}"
            
        # Check if it's an image file
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif'}
        if path.suffix.lower() not in valid_extensions:
            return False, f"File is not a supported image format: {path.suffix}"
            
        # Check file size
        try:
            if path.stat().st_size == 0:
                return False, f"Image file is empty: {path}"
        except Exception as e:
            return False, f"Failed to read file: {str(e)}"
            
        return True, None
        
    def _format_recognition_result(self, result: RecognitionResult) -> Dict[str, Any]:
        """
        Format a RecognitionResult for output.
        
        Args:
            result: RecognitionResult object
            
        Returns:
            Dict[str, Any]: Formatted recognition data
        """
        return {
            "shape": result.shape.value if hasattr(result.shape, 'value') else str(result.shape),
            "confidence": result.confidence,
            "features": {
                "area": result.features.area,
                "perimeter": result.features.perimeter,
                "vertices": result.features.vertices,
                "width": result.features.width,
                "height": result.features.height,
                "aspect_ratio": result.features.aspect_ratio,
                "circularity": result.features.circularity,
                "extent": result.features.extent
            }
        }
        
    def _format_geometry_result(self, result: GeometryResult) -> Dict[str, Any]:
        """
        Format a GeometryResult for output.
        
        Args:
            result: GeometryResult object
            
        Returns:
            Dict[str, Any]: Formatted geometry data
        """
        return {
            "type": result.shape,
            "dimensions": result.dimensions,
            "vertices": result.vertices_count,
            "faces": result.faces_count,
            "watertight": result.is_watertight,
            "success": result.success
        }
        
    def _format_stl_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format STL export result for output.
        
        Args:
            result: STL export result dictionary
            
        Returns:
            Dict[str, Any]: Formatted STL data
        """
        return {
            "path": result.get("path"),
            "filename": result.get("filename"),
            "size_bytes": result.get("size_bytes", 0),
            "vertices": result.get("vertices", 0),
            "faces": result.get("faces", 0),
            "watertight": result.get("watertight", False),
            "success": result.get("success", False)
        }
        
    def process(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process an image through the complete 2D-to-3D pipeline.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dict[str, Any]: Complete pipeline results
        """
        start_time = time.time()
        self._logger.info(f"Pipeline started for: {image_path}")
        
        # Initialize result structure
        result = {
            "success": False,
            "input": {"image_path": str(image_path)},
            "recognition": None,
            "geometry": None,
            # Runtime-only mesh for the UI viewport. STL/export metadata stays
            # in the serializable geometry and stl fields.
            "mesh": None,
            "stl": None,
            "stage": "input",
            "error": None,
            "processing_time": 0.0
        }
        
        try:
            # ============================================================
            # STEP 1: Validate input
            # ============================================================
            self._logger.info("STEP 1: Validating input...")
            is_valid, error = self._validate_input(image_path)
            
            if not is_valid:
                result["stage"] = "input"
                result["error"] = error
                self._logger.error(f"Input validation failed: {error}")
                result["processing_time"] = time.time() - start_time
                return result
                
            # ============================================================
            # STEP 2: Shape recognition
            # ============================================================
            self._logger.info("STEP 2: Performing shape recognition...")
            result["stage"] = "recognition"
            
            try:
                recognition_result = self.shape_recognizer.recognize(image_path)
            except Exception as e:
                result["error"] = f"Recognition failed: {str(e)}"
                self._logger.error(result["error"])
                result["processing_time"] = time.time() - start_time
                return result
                
            # Format recognition result
            result["recognition"] = self._format_recognition_result(recognition_result)
            shape_name = result["recognition"]["shape"]
            confidence = result["recognition"]["confidence"]
            
            self._logger.info(f"Detected shape: {shape_name} (confidence: {confidence:.2f})")

            if recognition_result.error:
                result["error"] = f"Recognition failed: {recognition_result.error}"
                self._logger.error(result["error"])
                return result
            
            # ============================================================
            # STEP 3: Confidence check
            # ============================================================
            self._logger.info("STEP 3: Checking confidence...")
            result["stage"] = "confidence"
            
            if confidence < self.confidence_threshold:
                result["error"] = (
                    f"Recognition confidence ({confidence:.2f}) below threshold "
                    f"({self.confidence_threshold:.2f})"
                )
                self._logger.warning(result["error"])
                result["processing_time"] = time.time() - start_time
                return result
                
            self._logger.info(f"Confidence accepted: {confidence:.2f} >= {self.confidence_threshold:.2f}")
            
            # ============================================================
            # STEP 4: Generate geometry
            # ============================================================
            self._logger.info("STEP 4: Generating 3D geometry...")
            result["stage"] = "geometry"
            
            try:
                geometry_result = self.geometry_generator.generate(recognition_result)
            except Exception as e:
                result["error"] = f"Geometry generation failed: {str(e)}"
                self._logger.error(result["error"])
                result["processing_time"] = time.time() - start_time
                return result
                
            result["geometry"] = self._format_geometry_result(geometry_result)
            
            if not geometry_result.success:
                result["error"] = geometry_result.error or "Geometry generation failed"
                self._logger.error(result["error"])
                result["processing_time"] = time.time() - start_time
                return result

            result["mesh"] = geometry_result.geometry
                
            self._logger.info(f"Geometry generated: {geometry_result.shape}, "
                            f"vertices: {geometry_result.vertices_count}, "
                            f"faces: {geometry_result.faces_count}")
            
            # ============================================================
            # STEP 5: Validate geometry
            # ============================================================
            self._logger.info("STEP 5: Validating geometry...")
            result["stage"] = "validation"
            
            if geometry_result.geometry is None:
                result["error"] = "Generated geometry is None"
                self._logger.error(result["error"])
                result["processing_time"] = time.time() - start_time
                return result
                
            # Additional validation using geometry generator
            try:
                is_valid = self.geometry_generator.validate_mesh(geometry_result.geometry)
                if not is_valid:
                    result["error"] = "Mesh validation failed"
                    self._logger.error("Mesh validation failed; STL export cancelled")
                    result["processing_time"] = time.time() - start_time
                    return result
            except Exception as e:
                result["error"] = f"Mesh validation failed: {str(e)}"
                self._logger.error(result["error"])
                result["processing_time"] = time.time() - start_time
                return result
                
            # ============================================================
            # STEP 6: Export STL
            # ============================================================
            self._logger.info("STEP 6: Exporting STL...")
            result["stage"] = "export"
            
            try:
                stl_result = self.stl_exporter.export(
                    mesh=geometry_result.geometry,
                    shape_name=shape_name,
                    overwrite=False,
                    repair=True,
                    verify=True
                )
            except Exception as e:
                result["error"] = f"STL export failed: {str(e)}"
                self._logger.error(result["error"])
                result["processing_time"] = time.time() - start_time
                return result
                
            result["stl"] = self._format_stl_result(stl_result)
            
            if not stl_result.get("success", False):
                result["error"] = stl_result.get("error", "STL export failed")
                self._logger.error(result["error"])
                result["processing_time"] = time.time() - start_time
                return result
                
            self._logger.info(f"STL exported: {stl_result.get('filename')} "
                            f"({stl_result.get('size_bytes', 0)} bytes)")
            
            # ============================================================
            # STEP 7: Verify result
            # ============================================================
            self._logger.info("STEP 7: Verifying final result...")
            result["stage"] = "verification"
            
            # Check final STL
            stl_path = stl_result.get("path")
            if stl_path:
                path_obj = Path(stl_path)
                if not path_obj.exists():
                    result["error"] = "STL file not found after export"
                    self._logger.error(result["error"])
                    result["processing_time"] = time.time() - start_time
                    return result
                    
                if path_obj.stat().st_size == 0:
                    result["error"] = "STL file is empty"
                    self._logger.error(result["error"])
                    result["processing_time"] = time.time() - start_time
                    return result
                    
            # ============================================================
            # COMPLETE: Success
            # ============================================================
            result["success"] = True
            result["stage"] = "complete"
            result["error"] = None
            
            self._logger.info(f"Pipeline completed successfully for: {image_path}")
            
        except Exception as e:
            # Catch any unexpected exceptions
            result["error"] = f"Unexpected error: {str(e)}"
            result["stage"] = "error"
            self._logger.error(f"Pipeline failed with unexpected error: {str(e)}")
            import traceback
            self._logger.error(traceback.format_exc())
            
        finally:
            result["processing_time"] = time.time() - start_time
            self._logger.info(f"Pipeline finished in {result['processing_time']:.2f} seconds")
            
        return result
        
    def process_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """
        Process multiple images through the pipeline.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List[Dict[str, Any]]: List of pipeline results
        """
        results = []
        total = len(image_paths)
        
        self._logger.info(f"Processing batch of {total} images...")
        
        for i, path in enumerate(image_paths, 1):
            self._logger.info(f"Processing image {i}/{total}: {path}")
            result = self.process(path)
            results.append(result)
            
        self._logger.info(f"Batch processing complete: {sum(1 for r in results if r['success'])} successful")
        return results
        
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline configuration.
        
        Returns:
            Dict[str, Any]: Pipeline information
        """
        return {
            "confidence_threshold": self.confidence_threshold,
            "output_dir": str(self.output_dir),
            "pixels_per_mm": self.pixels_per_mm,
            "default_height": self.default_height,
            "supported_shapes": self.shape_recognizer.get_supported_shapes(),
            "modules": {
                "shape_recognizer": "ShapeRecognizer",
                "geometry_generator": "GeometryGenerator",
                "stl_exporter": "STLExporter"
            }
        }
        
    def __repr__(self) -> str:
        """
        String representation of the ModelPipeline.
        
        Returns:
            str: Human-readable representation
        """
        return (
            f"ModelPipeline(threshold={self.confidence_threshold}, "
            f"scale={self.pixels_per_mm}px/mm, height={self.default_height}mm)"
        )


def test_pipeline():
    """
    Comprehensive test function for the ModelPipeline.
    """
    print("=" * 70)
    print("MODEL PIPELINE TEST")
    print("=" * 70)
    
    # Create pipeline
    pipeline = ModelPipeline(
        confidence_threshold=0.60,
        output_dir="test_models"
    )
    
    print("\nPipeline Configuration:")
    print("-" * 50)
    info = pipeline.get_pipeline_info()
    print(f"Confidence threshold: {info['confidence_threshold']:.2f}")
    print(f"Output directory: {info['output_dir']}")
    print(f"Pixel-to-mm scale: {info['pixels_per_mm']} px/mm")
    print(f"Default extrusion height: {info['default_height']} mm")
    print(f"Supported shapes: {', '.join(info['supported_shapes'])}")
    
    # Test images - user can add their own test images
    test_dir = Path("test_data")
    test_images = []
    
    # Look for test images
    for shape in ["circle", "square", "rectangle", "triangle", "ellipse", "line"]:
        for ext in [".png", ".jpg", ".bmp"]:
            path = test_dir / f"{shape}{ext}"
            if path.exists():
                test_images.append(path)
                break
                
    if not test_images:
        print("\n" + "=" * 70)
        print("No test images found!")
        print("=" * 70)
        print("\nTo test the pipeline, create test images in the 'test_data' folder:")
        print("  test_data/circle.png")
        print("  test_data/square.png")
        print("  test_data/rectangle.png")
        print("  test_data/triangle.png")
        print("  test_data/ellipse.png")
        print("  test_data/line.png")
        print("\nYou can also test with a single image:")
        print("  pipeline.process('test_data/my_shape.png')")
        print("=" * 70)
        return
        
    print("\n" + "=" * 70)
    print("RUNNING TESTS")
    print("=" * 70)
    
    for img_path in test_images:
        print(f"\n📷 Processing: {img_path.name}")
        print("-" * 50)
        
        result = pipeline.process(img_path)
        
        print(f"✅ Success: {result['success']}")
        print(f"⏱️ Time: {result['processing_time']:.2f} seconds")
        print(f"📍 Stage: {result['stage']}")
        
        if result['success']:
            if result.get('recognition'):
                print(f"\n🔍 Recognition:")
                print(f"   Shape: {result['recognition'].get('shape', 'Unknown')}")
                print(f"   Confidence: {result['recognition'].get('confidence', 0):.2f}")
                
            if result.get('geometry'):
                print(f"\n📐 Geometry:")
                print(f"   Type: {result['geometry'].get('type', 'Unknown')}")
                print(f"   Vertices: {result['geometry'].get('vertices', 0):,}")
                print(f"   Faces: {result['geometry'].get('faces', 0):,}")
                print(f"   Watertight: {result['geometry'].get('watertight', False)}")
                
                if result['geometry'].get('dimensions'):
                    dims = result['geometry']['dimensions']
                    dims_str = ", ".join(f"{k}={v:.1f}" for k, v in dims.items())
                    print(f"   Dimensions: {dims_str}")
                    
            if result.get('stl'):
                print(f"\n💾 STL:")
                print(f"   File: {result['stl'].get('filename', 'Unknown')}")
                print(f"   Size: {result['stl'].get('size_bytes', 0):,} bytes")
                print(f"   Vertices: {result['stl'].get('vertices', 0):,}")
                print(f"   Faces: {result['stl'].get('faces', 0):,}")
        else:
            print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
            
    print("\n" + "=" * 70)
    print("TEST COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Exported STL files are in: {pipeline.output_dir.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    test_pipeline()
