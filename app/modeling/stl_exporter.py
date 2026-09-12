"""
stl_exporter.py - STL Export Module for AI Smart Blackboard

This module provides STL export functionality for 3D geometries generated
from the Smart Blackboard. It handles mesh validation, file generation,
and export verification for 3D printing compatibility.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Union, Tuple
import logging
import numpy as np

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh not installed. Install with: pip install trimesh")


class STLExporter:
    """
    Professional STL exporter for 3D printable geometries.
    
    This class handles the complete STL export pipeline including:
    - Mesh validation and repair
    - Filename generation
    - Directory management
    - Export verification
    - Error handling
    
    The exported STL files are compatible with standard 3D printing
    slicers including Cura, PrusaSlicer, Bambu Studio, and OrcaSlicer.
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path] = "app/data/models",
        default_ext: str = ".stl",
        binary_format: bool = True
    ) -> None:
        """
        Initialize the STLExporter with configuration.
        
        Args:
            output_dir: Directory for exported STL files
            default_ext: Default file extension
            binary_format: Use binary STL format (True) or ASCII (False)
        """
        self.output_dir = Path(output_dir)
        self.default_ext = default_ext if default_ext.startswith('.') else f".{default_ext}"
        self.binary_format = binary_format
        self._last_exported_path: Optional[Path] = None
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self._logger = logging.getLogger(__name__)
        
        if not TRIMESH_AVAILABLE:
            self._logger.error("trimesh is not installed. Please install with: pip install trimesh")
            
        # Ensure output directory exists
        self.ensure_output_directory()
        
    def ensure_output_directory(self) -> bool:
        """
        Create the output directory if it does not exist.
        
        Returns:
            bool: True if directory exists or was created successfully
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._logger.info(f"Output directory ready: {self.output_dir}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to create output directory: {str(e)}")
            return False
            
    def get_output_directory(self) -> Path:
        """
        Get the output directory path.
        
        Returns:
            Path: The output directory
        """
        return self.output_dir
        
    def get_last_exported_path(self) -> Optional[Path]:
        """
        Get the path of the most recently exported STL file.
        
        Returns:
            Optional[Path]: Path to the last exported file, or None
        """
        return self._last_exported_path
        
    def generate_filename(
        self,
        shape_name: str = "shape",
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Generate a unique filename for export.
        
        Args:
            shape_name: Name of the shape (e.g., "circle", "cube")
            timestamp: Optional timestamp, defaults to current time
            
        Returns:
            str: Generated filename with .stl extension
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{shape_name.lower()}_{timestamp_str}{self.default_ext}"
        
        return filename
        
    def validate_mesh(self, mesh: 'trimesh.Trimesh', repair: bool = False) -> Tuple[bool, str]:
        """
        Validate a trimesh mesh for STL export.
        
        Args:
            mesh: trimesh.Trimesh object to validate
            repair: Attempt to repair the mesh if issues are found
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        # Check if mesh exists
        if mesh is None:
            return False, "Mesh is None"
            
        # Check if mesh is trimesh object
        if not isinstance(mesh, trimesh.Trimesh):
            return False, f"Mesh is not a trimesh.Trimesh object (got {type(mesh)})"
            
        # Check for vertices
        if not hasattr(mesh, 'vertices') or mesh.vertices is None:
            return False, "Mesh has no vertices"
            
        if len(mesh.vertices) == 0:
            return False, "Mesh has empty vertices"
            
        # Check for faces
        if not hasattr(mesh, 'faces') or mesh.faces is None:
            return False, "Mesh has no faces"
            
        if len(mesh.faces) == 0:
            return False, "Mesh has empty faces"
            
        # Check for NaN or infinite values
        if not np.isfinite(mesh.vertices).all():
            if repair:
                self._logger.warning("Attempting to repair NaN/infinite values...")
                mesh.vertices = np.nan_to_num(mesh.vertices, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                return False, "Mesh contains NaN or infinite values"
                
        # Check for valid face indices
        max_vertex = len(mesh.vertices) - 1
        if mesh.faces.max() > max_vertex or mesh.faces.min() < 0:
            if repair:
                self._logger.warning("Attempting to repair invalid face indices...")
                mesh.faces = np.clip(mesh.faces, 0, max_vertex)
            else:
                return False, "Mesh has invalid face indices"
                
        # Check bounds
        try:
            bounds = mesh.bounds
            if np.any(np.isinf(bounds)) or np.any(np.isnan(bounds)):
                return False, "Mesh has invalid bounds"
                
            extents = mesh.extents
            if np.any(extents <= 0):
                if repair:
                    self._logger.warning("Attempting to repair zero dimensions...")
                else:
                    return False, "Mesh has zero or negative dimensions"
                    
        except Exception as e:
            return False, f"Failed to compute mesh bounds: {str(e)}"
            
        # Check watertight status
        if not mesh.is_watertight:
            self._logger.warning("Mesh is not watertight - may cause slicing issues")
            
        return True, "Mesh is valid"
        
    def repair_mesh(self, mesh: 'trimesh.Trimesh') -> 'trimesh.Trimesh':
        """
        Attempt to repair common mesh issues.
        
        Args:
            mesh: trimesh.Trimesh object to repair
            
        Returns:
            trimesh.Trimesh: Repaired mesh
        """
        if not TRIMESH_AVAILABLE:
            return mesh
            
        try:
            # Remove duplicate vertices
            mesh.merge_vertices()
            
            # Remove duplicate faces
            mesh = mesh.unique()
            
            # Remove unreferenced vertices
            mesh.remove_unreferenced_vertices()
            
            # Fix normals
            if not mesh.is_watertight:
                try:
                    mesh.fix_normals()
                except Exception as e:
                    self._logger.warning(f"Failed to fix normals: {str(e)}")
                    
            # Fill holes if possible
            if not mesh.is_watertight:
                try:
                    mesh.fill_holes()
                except Exception as e:
                    self._logger.warning(f"Failed to fill holes: {str(e)}")
                    
            self._logger.info("Mesh repair completed")
            return mesh
            
        except Exception as e:
            self._logger.warning(f"Mesh repair failed: {str(e)}")
            return mesh
            
    def _find_unique_filename(self, filepath: Path) -> Path:
        """
        Find a unique filename if the requested file already exists.
        
        Args:
            filepath: Requested filepath
            
        Returns:
            Path: Unique filepath
        """
        if not filepath.exists():
            return filepath
            
        stem = filepath.stem
        suffix = filepath.suffix
        directory = filepath.parent
        
        counter = 1
        while True:
            new_stem = f"{stem}_{counter}"
            new_path = directory / f"{new_stem}{suffix}"
            if not new_path.exists():
                return new_path
            counter += 1
            
    def verify_export(self, filepath: Path) -> Dict[str, Any]:
        """
        Verify that an exported STL file is valid.
        
        Args:
            filepath: Path to the exported STL file
            
        Returns:
            Dict[str, Any]: Verification results
        """
        result = {
            "exists": False,
            "size_bytes": 0,
            "valid_stl": False,
            "vertices": 0,
            "faces": 0,
            "watertight": False,
            "error": None
        }
        
        try:
            # Check if file exists
            if not filepath.exists():
                result["error"] = "File does not exist"
                return result
                
            result["exists"] = True
            result["size_bytes"] = filepath.stat().st_size
            
            # Check file size
            if result["size_bytes"] == 0:
                result["error"] = "File is empty"
                return result
                
            # Reload the STL to verify
            loaded = trimesh.load(filepath)
            
            if loaded is None:
                result["error"] = "Failed to load STL file"
                return result
                
            if not isinstance(loaded, trimesh.Trimesh):
                result["error"] = "Loaded file is not a trimesh object"
                return result
                
            result["valid_stl"] = True
            result["vertices"] = len(loaded.vertices)
            result["faces"] = len(loaded.faces)
            result["watertight"] = loaded.is_watertight
            
            return result
            
        except Exception as e:
            result["error"] = str(e)
            return result
            
    def export(
        self,
        mesh: 'trimesh.Trimesh',
        filename: Optional[str] = None,
        shape_name: str = "shape",
        overwrite: bool = False,
        repair: bool = True,
        verify: bool = True
    ) -> Dict[str, Any]:
        """
        Export a trimesh mesh to an STL file.
        
        Args:
            mesh: trimesh.Trimesh object to export
            filename: Optional custom filename
            shape_name: Name of the shape for filename generation
            overwrite: Overwrite existing file
            repair: Attempt to repair mesh issues
            verify: Verify the exported file
            
        Returns:
            Dict[str, Any]: Export results with metadata
        """
        result = {
            "success": False,
            "path": None,
            "filename": None,
            "vertices": 0,
            "faces": 0,
            "watertight": False,
            "size_bytes": 0,
            "error": None
        }
        
        try:
            # Check if trimesh is available
            if not TRIMESH_AVAILABLE:
                result["error"] = "trimesh is not installed"
                self._logger.error(result["error"])
                return result
                
            # Validate mesh
            is_valid, message = self.validate_mesh(mesh, repair=repair)
            
            if not is_valid and not repair:
                result["error"] = f"Mesh validation failed: {message}"
                self._logger.error(result["error"])
                return result
                
            # Repair if needed and requested
            if repair and not is_valid:
                self._logger.info("Attempting mesh repair...")
                mesh = self.repair_mesh(mesh)
                
                # Re-validate after repair
                is_valid, message = self.validate_mesh(mesh, repair=False)
                if not is_valid:
                    result["error"] = f"Mesh repair failed: {message}"
                    self._logger.error(result["error"])
                    return result
                    
            # Generate filename
            if filename:
                # Ensure .stl extension
                if not filename.lower().endswith('.stl'):
                    filename = f"{filename}.stl"
                filepath = self.output_dir / filename
            else:
                filename = self.generate_filename(shape_name)
                filepath = self.output_dir / filename
                
            # Handle overwrite
            if not overwrite and filepath.exists():
                filepath = self._find_unique_filename(filepath)
                filename = filepath.name
                
            # Ensure output directory exists
            if not self.ensure_output_directory():
                result["error"] = "Failed to create output directory"
                return result
                
            # Export STL - FIXED: Use file_type without encoding parameter
            self._logger.info(f"Exporting STL to: {filepath}")
            
            try:
                # Correct trimesh export API
                mesh.export(
                    str(filepath),
                    file_type='stl'
                )
            except Exception as e:
                result["error"] = f"Export failed: {str(e)}"
                self._logger.error(result["error"])
                return result
                
            # Verify export
            if verify:
                verification = self.verify_export(filepath)
                if not verification["valid_stl"]:
                    result["error"] = f"Verification failed: {verification.get('error', 'Unknown error')}"
                    self._logger.error(result["error"])
                    return result
                    
                result["vertices"] = verification["vertices"]
                result["faces"] = verification["faces"]
                result["watertight"] = verification["watertight"]
                result["size_bytes"] = verification["size_bytes"]
                
            # Store last exported path
            self._last_exported_path = filepath
            
            # Build success result
            result["success"] = True
            result["path"] = str(filepath)
            result["filename"] = filename
            result["watertight"] = mesh.is_watertight
            
            if not result.get("vertices"):
                result["vertices"] = len(mesh.vertices)
                result["faces"] = len(mesh.faces)
                result["size_bytes"] = filepath.stat().st_size if filepath.exists() else 0
                
            self._logger.info(f"Export successful: {filepath}")
            return result
            
        except Exception as e:
            result["error"] = str(e)
            self._logger.error(f"Export failed: {str(e)}")
            return result
            
    def export_mesh(
        self,
        mesh: 'trimesh.Trimesh',
        filename: Optional[str] = None,
        shape_name: str = "shape",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Alias for export() method.
        
        Args:
            mesh: trimesh.Trimesh object to export
            filename: Optional custom filename
            shape_name: Name of the shape for filename generation
            **kwargs: Additional arguments for export()
            
        Returns:
            Dict[str, Any]: Export results
        """
        return self.export(mesh, filename, shape_name, **kwargs)
        
    def export_all_meshes(
        self,
        meshes: Dict[str, 'trimesh.Trimesh'],
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Export multiple meshes.
        
        Args:
            meshes: Dictionary mapping shape names to trimesh objects
            **kwargs: Additional arguments for export()
            
        Returns:
            Dict[str, Dict[str, Any]]: Export results for each mesh
        """
        results = {}
        for shape_name, mesh in meshes.items():
            results[shape_name] = self.export(mesh, shape_name=shape_name, **kwargs)
        return results
        
    def get_export_info(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about an exported STL file.
        
        Args:
            filepath: Path to the STL file
            
        Returns:
            Dict[str, Any]: File information
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            return {"error": f"File not found: {filepath}"}
            
        try:
            loaded = trimesh.load(filepath)
            
            return {
                "path": str(filepath),
                "filename": filepath.name,
                "size_bytes": filepath.stat().st_size,
                "vertices": len(loaded.vertices),
                "faces": len(loaded.faces),
                "watertight": loaded.is_watertight,
                "volume": loaded.volume,
                "bounds": loaded.bounds.tolist() if hasattr(loaded, 'bounds') else None
            }
        except Exception as e:
            return {"error": f"Failed to load file: {str(e)}"}
            
    def __repr__(self) -> str:
        """
        String representation of the STLExporter.
        
        Returns:
            str: Human-readable representation
        """
        return (
            f"STLExporter(output_dir='{self.output_dir}', "
            f"format={'binary' if self.binary_format else 'ascii'})"
        )


def test_stl_exporter():
    """
    Comprehensive test function for the STLExporter.
    """
    print("=" * 70)
    print("STL EXPORTER TEST")
    print("=" * 70)
    
    if not TRIMESH_AVAILABLE:
        print("trimesh is not installed. Please install with: pip install trimesh")
        return
        
    # Create exporter
    exporter = STLExporter(output_dir="test_models")
    
    # Create test shapes
    test_shapes = {
        "cylinder": trimesh.creation.cylinder(radius=10, height=20, segments=32),
        "cube": trimesh.creation.box(extents=[20, 20, 20]),
        "box": trimesh.creation.box(extents=[30, 20, 15]),
        "sphere": trimesh.creation.icosphere(radius=10, subdivisions=2)
    }
    
    print("\n" + "=" * 70)
    print("TESTING SHAPE EXPORTS")
    print("=" * 70)
    
    for shape_name, mesh in test_shapes.items():
        print(f"\n🔄 Exporting: {shape_name.capitalize()}")
        print("-" * 50)
        
        # Export with verification
        result = exporter.export(
            mesh,
            shape_name=shape_name,
            overwrite=True,
            repair=True,
            verify=True
        )
        
        print(f"✅ Success: {result['success']}")
        
        if result['success']:
            print(f"📁 File: {result['path']}")
            print(f"📦 Size: {result.get('size_bytes', 0):,} bytes")
            print(f"🔺 Vertices: {result.get('vertices', 0):,}")
            print(f"▣ Faces: {result.get('faces', 0):,}")
            print(f"💧 Watertight: {result.get('watertight', False)}")
            
            # Also get detailed info
            info = exporter.get_export_info(result['path'])
            if 'error' not in info:
                print(f"📐 Volume: {info.get('volume', 0):.2f} mm³")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            
    print("\n" + "=" * 70)
    print("TESTING FILENAME GENERATION")
    print("=" * 70)
    
    # Test filename generation
    filenames = [
        exporter.generate_filename("circle"),
        exporter.generate_filename("square"),
        exporter.generate_filename("triangle"),
        exporter.generate_filename("custom_shape")
    ]
    
    for i, filename in enumerate(filenames, 1):
        print(f"{i}. {filename}")
        
    print("\n" + "=" * 70)
    print("TESTING OVERWRITE SAFETY")
    print("=" * 70)
    
    # Test duplicate filename handling
    mesh = trimesh.creation.box(extents=[10, 10, 10])
    
    # First export
    result1 = exporter.export(mesh, filename="test_duplicate.stl", overwrite=False)
    print(f"First export: {result1['filename']}")
    
    # Second export (should auto-increment)
    result2 = exporter.export(mesh, filename="test_duplicate.stl", overwrite=False)
    print(f"Second export: {result2['filename']}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE!")
    print("=" * 70)
    print(f"📁 Exported files are in: {exporter.output_dir.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    test_stl_exporter()