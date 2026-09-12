"""
geometry_generator.py - 3D Geometry Generation for AI Smart Blackboard

This module converts recognized 2D shapes into 3D printable geometry.
It takes the output from ShapeRecognizer and generates appropriate 3D meshes
for shapes including circles, squares, rectangles, triangles, ellipses, and lines.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import dataclass, field
import logging
from pathlib import Path

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh not installed. Install with: pip install trimesh")

import sys


@dataclass
class GeometryConfig:
    """Configuration for geometry generation."""
    default_height: float = 20.0  # Default extrusion height in mm
    default_depth: float = 20.0   # Default depth for cuboids
    pixels_per_mm: float = 5.0    # Scale factor: pixels to millimeters
    min_confidence: float = 0.70  # Minimum confidence for generation
    min_dimension: float = 1.0    # Minimum dimension in mm
    max_dimension: float = 500.0  # Maximum dimension in mm


@dataclass
class GeometryResult:
    """Container for geometry generation results."""
    success: bool
    shape: str
    geometry: Optional['trimesh.Trimesh'] = None
    dimensions: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    error: Optional[str] = None
    vertices_count: int = 0
    faces_count: int = 0
    is_watertight: bool = False


class GeometryGenerator:
    """
    Converts recognized 2D shapes into 3D printable geometry.
    
    This class takes the output from ShapeRecognizer and generates
    appropriate 3D meshes for various shapes using trimesh.
    
    Supported shapes:
    - Circle → Cylinder
    - Square → Cube
    - Rectangle → Cuboid
    - Triangle → Triangular Prism
    - Ellipse → Elliptical Cylinder
    - Line → Thin Rectangular Prism
    
    The generated geometry is suitable for STL export and 3D printing.
    """
    
    def __init__(self, config: Optional[GeometryConfig] = None) -> None:
        """
        Initialize the GeometryGenerator with configuration.
        
        Args:
            config: Optional configuration override
        """
        self.config = config or GeometryConfig()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self._logger = logging.getLogger(__name__)
        
        if not TRIMESH_AVAILABLE:
            self._logger.error("trimesh is not installed. Please install with: pip install trimesh")
            
    def set_scale(self, pixels_per_mm: float) -> None:
        """
        Set the pixel-to-millimeter scale factor.
        
        Args:
            pixels_per_mm: Number of pixels per millimeter
        """
        if pixels_per_mm <= 0:
            raise ValueError("pixels_per_mm must be positive")
        self.config.pixels_per_mm = pixels_per_mm
        self._logger.info(f"Scale set to {pixels_per_mm} pixels/mm")
        
    def set_min_confidence(self, min_confidence: float) -> None:
        """
        Set the minimum confidence threshold for generation.
        
        Args:
            min_confidence: Minimum confidence value (0.0 to 1.0)
        """
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        self.config.min_confidence = min_confidence
        self._logger.info(f"Minimum confidence set to {min_confidence}")
        
    def _pixels_to_mm(self, pixels: float) -> float:
        """
        Convert pixels to millimeters using the configured scale.
        
        Args:
            pixels: Value in pixels
            
        Returns:
            float: Value in millimeters
        """
        return pixels / self.config.pixels_per_mm
        
    def _validate_dimension(self, value: float, name: str) -> float:
        """
        Validate a dimension value.
        
        Args:
            value: Dimension value to validate
            name: Name of the dimension for error messages
            
        Returns:
            float: Validated dimension
            
        Raises:
            ValueError: If dimension is invalid
        """
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        if value < self.config.min_dimension:
            self._logger.warning(f"{name} ({value:.2f}mm) is very small")
            return self.config.min_dimension
        if value > self.config.max_dimension:
            raise ValueError(f"{name} ({value:.2f}mm) exceeds maximum dimension {self.config.max_dimension}mm")
        return value
        
    def calculate_dimensions(self, features: Union[Dict[str, Any], Any]) -> Dict[str, float]:
        """
        Calculate real-world dimensions from pixel-based features.
        
        Args:
            features: Feature dictionary from ShapeRecognizer
            
        Returns:
            Dict[str, float]: Calculated dimensions in millimeters
        """
        # ShapeRecognizer exposes ShapeFeatures as a dataclass. Keep support
        # for dictionaries so this generator is usable independently too.
        get_feature = features.get if isinstance(features, dict) else lambda name, default=0: getattr(features, name, default)
        width_px = get_feature('width', 0)
        height_px = get_feature('height', 0)
        area_px = get_feature('area', 0)
        perimeter_px = get_feature('perimeter', 0)
        
        # Convert to millimeters
        width_mm = self._pixels_to_mm(width_px)
        height_mm = self._pixels_to_mm(height_px)
        area_mm2 = area_px / (self.config.pixels_per_mm ** 2)
        perimeter_mm = perimeter_px / self.config.pixels_per_mm
        
        return {
            'width': self._validate_dimension(width_mm, 'Width'),
            'height': self._validate_dimension(height_mm, 'Height'),
            'depth': self._validate_dimension(width_mm, 'Depth'),
            'area': area_mm2,
            'perimeter': perimeter_mm
        }
        
    def validate_mesh(self, mesh: 'trimesh.Trimesh') -> bool:
        """
        Validate a generated mesh.
        
        Args:
            mesh: trimesh object to validate
            
        Returns:
            bool: True if mesh is valid
        """
        if mesh is None:
            self._logger.error("Mesh is None")
            return False
            
        if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
            self._logger.error("Mesh has no vertices")
            return False
            
        if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
            self._logger.error("Mesh has no faces")
            return False
            
        # Check for NaN or infinite values
        if not np.isfinite(mesh.vertices).all():
            self._logger.error("Mesh contains NaN or infinite values")
            return False
            
        if not np.isfinite(mesh.face_normals).all():
            self._logger.error("Mesh contains invalid face normals")
            return False
        if not mesh.is_watertight:
            self._logger.error("Generated solid is not watertight")
            return False
        if not np.isfinite(mesh.volume) or mesh.volume <= 0:
            self._logger.error("Generated solid has non-positive volume")
            return False
        return True
            
    def create_cylinder(self, radius: float, height: float, segments: int = 32) -> 'trimesh.Trimesh':
        """
        Create a cylinder mesh.
        
        Args:
            radius: Radius of the cylinder in mm
            height: Height of the cylinder in mm
            segments: Number of segments for circular approximation
            
        Returns:
            trimesh.Trimesh: Cylinder mesh
        """
        if not TRIMESH_AVAILABLE:
            raise ImportError("trimesh is not installed")
            
        radius = self._validate_dimension(radius, 'Radius')
        height = self._validate_dimension(height, 'Height')
        
        # Create cylinder using trimesh
        mesh = trimesh.creation.cylinder(
            radius=radius,
            height=height,
            segments=segments
        )
        
        # Translate to sit on Z=0
        mesh.apply_translation([0, 0, height / 2])
        
        return mesh
        
    def create_cube(self, side: float) -> 'trimesh.Trimesh':
        """
        Create a cube mesh.
        
        Args:
            side: Side length in mm
            
        Returns:
            trimesh.Trimesh: Cube mesh
        """
        if not TRIMESH_AVAILABLE:
            raise ImportError("trimesh is not installed")
            
        side = self._validate_dimension(side, 'Side')
        
        # Create box (cube) using trimesh
        mesh = trimesh.creation.box(extents=[side, side, side])
        
        # Translate to sit on Z=0
        mesh.apply_translation([0, 0, side / 2])
        
        return mesh
        
    def create_cuboid(self, width: float, depth: float, height: float) -> 'trimesh.Trimesh':
        """
        Create a cuboid (rectangular prism) mesh.
        
        Args:
            width: Width of the cuboid in mm
            depth: Depth of the cuboid in mm
            height: Height of the cuboid in mm
            
        Returns:
            trimesh.Trimesh: Cuboid mesh
        """
        if not TRIMESH_AVAILABLE:
            raise ImportError("trimesh is not installed")
            
        width = self._validate_dimension(width, 'Width')
        depth = self._validate_dimension(depth, 'Depth')
        height = self._validate_dimension(height, 'Height')
        
        # Create box using trimesh
        mesh = trimesh.creation.box(extents=[width, depth, height])
        
        # Translate to sit on Z=0
        mesh.apply_translation([0, 0, height / 2])
        
        return mesh
        
    def create_triangular_prism(
        self,
        base: float,
        height_triangle: float,
        extrusion_height: float
    ) -> 'trimesh.Trimesh':
        """
        Create a triangular prism mesh.
        
        Args:
            base: Base length of the triangle in mm
            height_triangle: Height of the triangle in mm
            extrusion_height: Extrusion height in mm
            
        Returns:
            trimesh.Trimesh: Triangular prism mesh
        """
        if not TRIMESH_AVAILABLE:
            raise ImportError("trimesh is not installed")
            
        base = self._validate_dimension(base, 'Base')
        height_triangle = self._validate_dimension(height_triangle, 'Triangle Height')
        extrusion_height = self._validate_dimension(extrusion_height, 'Extrusion Height')
        
        # Direct construction avoids optional polygon-triangulation backends
        # and guarantees a six-vertex watertight prism.
        bottom = np.array([
            [-base / 2, -height_triangle / 3, 0],
            [base / 2, -height_triangle / 3, 0],
            [0, 2 * height_triangle / 3, 0],
        ])
        vertices = np.vstack((bottom, bottom + [0, 0, extrusion_height]))
        faces = np.array([
            [0, 2, 1], [3, 4, 5],
            [0, 1, 4], [0, 4, 3],
            [1, 2, 5], [1, 5, 4],
            [2, 0, 3], [2, 3, 5],
        ])
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        
    def create_elliptical_cylinder(
        self,
        radius_x: float,
        radius_y: float,
        height: float,
        segments: int = 32
    ) -> 'trimesh.Trimesh':
        """
        Create an elliptical cylinder mesh.
        
        Args:
            radius_x: X-axis radius in mm
            radius_y: Y-axis radius in mm
            height: Height in mm
            segments: Number of segments for elliptical approximation
            
        Returns:
            trimesh.Trimesh: Elliptical cylinder mesh
        """
        if not TRIMESH_AVAILABLE:
            raise ImportError("trimesh is not installed")
            
        radius_x = self._validate_dimension(radius_x, 'X Radius')
        radius_y = self._validate_dimension(radius_y, 'Y Radius')
        height = self._validate_dimension(height, 'Height')
        
        # Create elliptical cylinder by scaling a circle
        # Create a cylinder
        mesh = trimesh.creation.cylinder(
            radius=1.0,
            height=height,
            segments=segments
        )
        
        # Apply elliptical scaling
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = radius_x
        scale_matrix[1, 1] = radius_y
        
        mesh.apply_transform(scale_matrix)
        
        # Translate to sit on Z=0
        mesh.apply_translation([0, 0, height / 2])
        
        return mesh
        
    def create_line_prism(
        self,
        length: float,
        width: float,
        height: float
    ) -> 'trimesh.Trimesh':
        """
        Create a thin rectangular prism for line shapes.
        
        Args:
            length: Length of the line in mm
            width: Width of the line in mm (extrusion width)
            height: Height of the line in mm (extrusion depth)
            
        Returns:
            trimesh.Trimesh: Rectangular prism mesh
        """
        if not TRIMESH_AVAILABLE:
            raise ImportError("trimesh is not installed")
            
        length = self._validate_dimension(length, 'Length')
        width = self._validate_dimension(width, 'Width')
        height = self._validate_dimension(height, 'Height')
        
        # Create box (line prism)
        mesh = trimesh.creation.box(extents=[length, width, height])
        
        # Translate to sit on Z=0
        mesh.apply_translation([0, 0, height / 2])
        
        return mesh

    @staticmethod
    def _polygon_area(vertices: np.ndarray) -> float:
        return float(np.dot(vertices[:, 0], np.roll(vertices[:, 1], -1)) - np.dot(vertices[:, 1], np.roll(vertices[:, 0], -1))) / 2.0

    def create_polygon_prism(self, vertices: Union[List[Tuple[float, float]], np.ndarray], height: float) -> 'trimesh.Trimesh':
        """Extrude a simple 2D polygon into a watertight mesh using ear clipping.

        This deliberately avoids optional triangulation packages: every supported
        drawn polygon, including concave stars/hearts/crescents, uses the same
        deterministic prism construction.
        """
        if not TRIMESH_AVAILABLE:
            raise ImportError("trimesh is not installed")
        points = np.asarray(vertices, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError("Polygon extrusion requires at least three (x, y) vertices")
        height = self._validate_dimension(height, "Extrusion height")
        # Remove consecutive duplicate points, which would make degenerate ears.
        keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]
        points = points[keep]
        if len(points) < 3 or abs(self._polygon_area(points)) < 1e-8:
            raise ValueError("Polygon has zero area")
        if self._polygon_area(points) < 0:
            points = points[::-1]

        def cross(a, b, c):
            ab = b - a
            bc = c - b
            return float(ab[0] * bc[1] - ab[1] * bc[0])

        def contains(point, a, b, c):
            # Inclusive barycentric test; ears may not contain another vertex.
            values = [cross(a, b, point), cross(b, c, point), cross(c, a, point)]
            return min(values) >= -1e-9 or max(values) <= 1e-9

        indices = list(range(len(points)))
        triangles: List[Tuple[int, int, int]] = []
        while len(indices) > 3:
            ear_found = False
            for position, current in enumerate(indices):
                previous = indices[position - 1]
                following = indices[(position + 1) % len(indices)]
                a, b, c = points[previous], points[current], points[following]
                if cross(a, b, c) <= 1e-9:
                    continue
                others = [idx for idx in indices if idx not in (previous, current, following)]
                if any(contains(points[idx], a, b, c) for idx in others):
                    continue
                triangles.append((previous, current, following))
                indices.pop(position)
                ear_found = True
                break
            if not ear_found:
                raise ValueError("Could not triangulate polygon; outline is self-intersecting")
        triangles.append(tuple(indices))

        count = len(points)
        vertices_3d = np.vstack((np.column_stack((points, np.zeros(count))), np.column_stack((points, np.full(count, height)))))
        faces: List[Tuple[int, int, int]] = []
        for a, b, c in triangles:
            faces.extend(((c, b, a), (a + count, b + count, c + count)))
        for i in range(count):
            j = (i + 1) % count
            faces.extend(((i, j, j + count), (i, j + count, i + count)))
        return trimesh.Trimesh(vertices=vertices_3d, faces=np.asarray(faces), process=True)

    def _regular_polygon(self, sides: int, width: float, height: float) -> np.ndarray:
        angles = np.linspace(0, 2 * np.pi, sides, endpoint=False) + np.pi / 2
        return np.column_stack((width / 2 * np.cos(angles), height / 2 * np.sin(angles)))

    def _template_polygon(self, shape: str, width: float, height: float) -> np.ndarray:
        """Return normalized outlines for non-regular recognized 2D symbols."""
        templates = {
            "trapezoid": [(-.5, -.5), (.5, -.5), (.30, .5), (-.30, .5)],
            "parallelogram": [(-.5, -.5), (.25, -.5), (.5, .5), (-.25, .5)],
            "rhombus": [(0, -.5), (.5, 0), (0, .5), (-.5, 0)],
            "kite": [(0, -.5), (.38, 0), (0, .5), (-.22, 0)],
            "arrow": [(-.5, -.22), (.08, -.22), (.08, -.5), (.5, 0), (.08, .5), (.08, .22), (-.5, .22)],
            "cross": [(-.18, -.5), (.18, -.5), (.18, -.18), (.5, -.18), (.5, .18), (.18, .18), (.18, .5), (-.18, .5), (-.18, .18), (-.5, .18), (-.5, -.18), (-.18, -.18)],
            "star": [(0, -.5), (.112, -.155), (.476, -.155), (.182, .059), (.294, .405), (0, .19), (-.294, .405), (-.182, .059), (-.476, -.155), (-.112, -.155)],
            "heart": [(0, .48), (-.5, .02), (-.42, -.30), (-.22, -.5), (0, -.30), (.22, -.5), (.42, -.30), (.5, .02)],
        }
        if shape == "semicircle":
            angles = np.linspace(np.pi, 0, 20)
            points = np.column_stack((.5 * np.cos(angles), .5 * np.sin(angles)))
            # The arc endpoints already form the chord when the polygon closes.
            return points * [width, height]
        if shape == "quarter_circle":
            angles = np.linspace(0, np.pi / 2, 16)
            points = np.column_stack((.5 * np.cos(angles), .5 * np.sin(angles)))
            return np.vstack(([(0, 0)], points)) * [width, height]
        if shape == "crescent":
            outer = [(0.5 * np.cos(a), 0.5 * np.sin(a)) for a in np.linspace(np.pi / 2, 3 * np.pi / 2, 20)]
            inner = [(0.20 + .34 * np.cos(a), .34 * np.sin(a)) for a in np.linspace(3 * np.pi / 2, np.pi / 2, 20)]
            return np.asarray(outer + inner) * [width, height]
        return np.asarray(templates[shape]) * [width, height]
        
    def generate(self, shape_result: Any) -> GeometryResult:
        """
        Generate 3D geometry from a shape recognition result.
        
        Args:
            shape_result: Result from ShapeRecognizer containing:
                - shape: Shape type string
                - confidence: Confidence score
                - features: Geometric features dictionary
            
        Returns:
            GeometryResult: Generated geometry with metadata
        """
        try:
            # Validate input
            if not shape_result:
                return GeometryResult(
                    success=False,
                    shape="unknown",
                    error="No shape result provided"
                )
                
            if isinstance(shape_result, dict):
                raw_shape = shape_result.get('shape', 'unknown')
                confidence = shape_result.get('confidence', 0.0)
                features = shape_result.get('features')
            else:
                raw_shape = getattr(shape_result, 'shape', 'unknown')
                confidence = getattr(shape_result, 'confidence', 0.0)
                features = getattr(shape_result, 'features', None)
            shape = str(getattr(raw_shape, 'value', raw_shape)).strip().lower()
            confidence = float(confidence)
            
            # Check confidence
            if confidence < self.config.min_confidence:
                return GeometryResult(
                    success=False,
                    shape=shape,
                    confidence=confidence,
                    error=f"Low confidence ({confidence:.2f} < {self.config.min_confidence})"
                )
                
            # Check features
            if features is None:
                return GeometryResult(
                    success=False,
                    shape=shape,
                    confidence=confidence,
                    error="No features provided"
                )
                
            # Calculate dimensions
            dimensions = self.calculate_dimensions(features)
            
            # Generate geometry based on shape
            geometry = None
            dims_dict = {}
            
            if shape == 'circle':
                width = dimensions['width']
                radius = width / 2
                geometry = self.create_cylinder(
                    radius=radius,
                    height=self.config.default_height
                )
                dims_dict = {
                    'radius': radius,
                    'diameter': width,
                    'height': self.config.default_height
                }
                
            elif shape == 'square':
                side = dimensions['width']
                geometry = self.create_cube(side=side)
                dims_dict = {
                    'side': side
                }
                
            elif shape == 'rectangle':
                width = dimensions['width']
                depth = dimensions['height']
                height = self.config.default_height
                geometry = self.create_cuboid(
                    width=width,
                    depth=depth,
                    height=height
                )
                dims_dict = {
                    'width': width,
                    'depth': depth,
                    'height': height
                }
                
            elif shape == 'triangle':
                base = dimensions['width']
                height_triangle = dimensions['height']
                extrusion_height = self.config.default_height
                geometry = self.create_triangular_prism(
                    base=base,
                    height_triangle=height_triangle,
                    extrusion_height=extrusion_height
                )
                dims_dict = {
                    'base': base,
                    'triangle_height': height_triangle,
                    'extrusion_height': extrusion_height
                }

            elif shape in {'pentagon', 'hexagon', 'heptagon', 'octagon', 'nonagon', 'decagon', 'undecagon', 'dodecagon'}:
                sides = {'pentagon': 5, 'hexagon': 6, 'heptagon': 7, 'octagon': 8, 'nonagon': 9, 'decagon': 10, 'undecagon': 11, 'dodecagon': 12}[shape]
                width, depth = dimensions['width'], dimensions['height']
                geometry = self.create_polygon_prism(self._regular_polygon(sides, width, depth), self.config.default_height)
                dims_dict = {'sides': sides, 'width': width, 'depth': depth, 'height': self.config.default_height}

            elif shape in {'trapezoid', 'parallelogram', 'rhombus', 'kite', 'star', 'arrow', 'cross', 'heart', 'semicircle', 'quarter_circle', 'crescent'}:
                width, depth = dimensions['width'], dimensions['height']
                geometry = self.create_polygon_prism(self._template_polygon(shape, width, depth), self.config.default_height)
                dims_dict = {'width': width, 'depth': depth, 'height': self.config.default_height}
                
            elif shape == 'ellipse':
                width = dimensions['width']
                depth = dimensions['height']
                radius_x = width / 2
                radius_y = depth / 2
                height = self.config.default_height
                geometry = self.create_elliptical_cylinder(
                    radius_x=radius_x,
                    radius_y=radius_y,
                    height=height
                )
                dims_dict = {
                    'radius_x': radius_x,
                    'radius_y': radius_y,
                    'height': height
                }
                
            elif shape == 'line':
                length = dimensions['width']
                width = self.config.default_depth * 0.5
                height = self.config.default_height
                geometry = self.create_line_prism(
                    length=length,
                    width=width,
                    height=height
                )
                dims_dict = {
                    'length': length,
                    'width': width,
                    'height': height
                }
                
            else:
                return GeometryResult(
                    success=False,
                    shape=shape,
                    confidence=confidence,
                    error=f"Unsupported shape: {shape}"
                )
                
            # Validate generated geometry
            if geometry is None:
                return GeometryResult(
                    success=False,
                    shape=shape,
                    confidence=confidence,
                    error="Geometry generation failed"
                )
                
            if not self.validate_mesh(geometry):
                return GeometryResult(
                    success=False,
                    shape=shape,
                    confidence=confidence,
                    error="Generated mesh is invalid"
                )
                
            return GeometryResult(
                success=True,
                shape=shape,
                geometry=geometry,
                dimensions=dims_dict,
                confidence=confidence,
                vertices_count=len(geometry.vertices),
                faces_count=len(geometry.faces),
                is_watertight=geometry.is_watertight
            )
            
        except Exception as e:
            self._logger.error(f"Geometry generation failed: {str(e)}")
            return GeometryResult(
                success=False,
                shape="unknown",
                error=str(e)
            )
            
    def generate_batch(
        self,
        shape_results: List[Dict[str, Any]]
    ) -> List[GeometryResult]:
        """
        Generate geometry for multiple shape recognition results.
        
        Args:
            shape_results: List of shape recognition results
            
        Returns:
            List[GeometryResult]: List of geometry results
        """
        return [self.generate(result) for result in shape_results]
        
    def export_stl(self, geometry: 'trimesh.Trimesh', filepath: Union[str, Path]) -> bool:
        """
        Export geometry as STL file.
        
        Args:
            geometry: trimesh geometry object
            filepath: Path to save the STL file
            
        Returns:
            bool: True if export successful
        """
        try:
            if geometry is None:
                self._logger.error("No geometry to export")
                return False
                
            geometry.export(str(filepath), 'stl')
            self._logger.info(f"Exported STL to: {filepath}")
            return True
            
        except Exception as e:
            self._logger.error(f"STL export failed: {str(e)}")
            return False
            
    def __repr__(self) -> str:
        """
        String representation of the GeometryGenerator.
        
        Returns:
            str: Human-readable representation
        """
        return f"GeometryGenerator(scale={self.config.pixels_per_mm}px/mm, min_conf={self.config.min_confidence})"


def test_geometry_generator():
    """
    Test function for the GeometryGenerator.
    """
    print("=" * 60)
    print("Geometry Generator Test")
    print("=" * 60)
    
    if not TRIMESH_AVAILABLE:
        print("trimesh is not installed. Please install with: pip install trimesh")
        return
        
    # Create generator
    generator = GeometryGenerator()
    
    # Test shapes
    test_shapes = [
        {
            'shape': 'circle',
            'confidence': 0.94,
            'features': {'width': 150, 'height': 148, 'area': 17671}
        },
        {
            'shape': 'square',
            'confidence': 0.92,
            'features': {'width': 150, 'height': 148, 'area': 22200}
        },
        {
            'shape': 'rectangle',
            'confidence': 0.90,
            'features': {'width': 200, 'height': 120, 'area': 24000}
        },
        {
            'shape': 'triangle',
            'confidence': 0.88,
            'features': {'width': 160, 'height': 140, 'area': 11200}
        },
        {
            'shape': 'ellipse',
            'confidence': 0.85,
            'features': {'width': 180, 'height': 120, 'area': 16965}
        },
        {
            'shape': 'line',
            'confidence': 0.85,
            'features': {'width': 250, 'height': 10, 'area': 1200}
        }
    ]
    
    print("\nTesting shape generation...")
    print("-" * 60)
    
    for i, test in enumerate(test_shapes, 1):
        print(f"\nTest {i}: {test['shape'].capitalize()}")
        print(f"  Confidence: {test['confidence']:.2f}")
        print(f"  Features: {test['features']}")
        
        result = generator.generate(test)
        
        print(f"  Success: {result.success}")
        if result.success:
            print(f"  Vertices: {result.vertices_count}")
            print(f"  Faces: {result.faces_count}")
            print(f"  Watertight: {result.is_watertight}")
            print(f"  Dimensions: {result.dimensions}")
        else:
            print(f"  Error: {result.error}")
            
    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    test_geometry_generator()
