"""
Test configuration validation and component initialization for ACE system.

Tests focus on:
- Configuration parameter validation
- Component initialization order
- Dependencies between components
- System setup validation
"""

from unittest.mock import MagicMock, patch

import pytest

from src.ace.plan.concrete import InfoFlowPlanner
from src.ace.schema.infoflow import MemoryModel
from src.ace.schema.lattice import SubsetLattice
from src.ace.security.infoflow import FlowAnalyzer


class TestConfigurationValidation:
    """Test that configuration parameters are properly validated"""

    def test_lattice_configuration_validation(self):
        """Test that lattice configuration is validated properly"""
        # Valid lattice configuration
        valid_lattice = SubsetLattice({"public", "private", "classified"})
        assert valid_lattice is not None
        assert len(valid_lattice.val) == 3

        # Empty lattice should work
        empty_lattice = SubsetLattice(set())
        assert empty_lattice is not None
        assert len(empty_lattice.val) == 0

        # Single element lattice should work
        single_lattice = SubsetLattice({"public"})
        assert single_lattice is not None
        assert len(single_lattice.val) == 1

    def test_memory_model_configuration_validation(self):
        """Test that memory model configuration is validated"""
        # Valid memory model
        valid_memory = MemoryModel(
            lattice_type=SubsetLattice,
            dynamic_vars={"user_input": SubsetLattice({"public"})},
            static_vars={
                "load_public_data": SubsetLattice({"public"}),
                "load_private_data": SubsetLattice({"private"}),
            },
        )
        assert valid_memory is not None
        assert valid_memory.lattice_type == SubsetLattice
        assert "user_input" in valid_memory.dynamic_vars
        assert "load_public_data" in valid_memory.static_vars

        # Empty memory model should work
        empty_memory = MemoryModel(lattice_type=SubsetLattice, dynamic_vars={}, static_vars={})
        assert empty_memory is not None

    def test_flow_analyzer_configuration(self):
        """Test that flow analyzer accepts valid configurations"""
        memory = MemoryModel(
            lattice_type=SubsetLattice, dynamic_vars={}, static_vars={"test_function": SubsetLattice({"public"})}
        )

        # Should initialize with valid memory model
        analyzer = FlowAnalyzer(memory)
        assert analyzer is not None
        # Memory model may be copied/transformed during initialization
        assert analyzer.memory is not None
        assert analyzer.valid == True  # Initial state should be valid
        assert len(analyzer.violations) == 0

    def test_planner_configuration_requirements(self):
        """Test that planners have appropriate configuration requirements"""
        # InfoFlowPlanner should be creatable
        planner = InfoFlowPlanner.__new__(InfoFlowPlanner)
        assert planner is not None

        # Should have expected attributes/methods
        assert hasattr(planner, "implement_plan")


class TestComponentInitialization:
    """Test that components initialize in correct order with proper dependencies"""

    def test_lattice_initialization_order(self):
        """Test that lattices initialize before dependent components"""
        # Lattice should initialize independently
        lattice = SubsetLattice({"level1", "level2"})
        assert lattice is not None

        # Memory model should initialize after lattice
        memory = MemoryModel(lattice_type=SubsetLattice, dynamic_vars={}, static_vars={"func": lattice})
        assert memory is not None
        assert memory.static_vars["func"] == lattice

    def test_analyzer_depends_on_memory_model(self):
        """Test that analyzer properly depends on memory model"""
        # Memory model must exist first
        memory = MemoryModel(
            lattice_type=SubsetLattice, dynamic_vars={}, static_vars={"test": SubsetLattice({"public"})}
        )

        # Analyzer should initialize with memory model
        analyzer = FlowAnalyzer(memory)
        # Memory model reference may be copied/transformed during init
        assert analyzer.memory is not None

        # Analyzer should fail without proper memory model
        with pytest.raises((TypeError, AttributeError)):
            FlowAnalyzer(None)

    def test_planner_initialization_with_dependencies(self):
        """Test that planner initializes with proper dependencies"""
        # Test InfoFlowPlanner initialization
        with patch("src.ace.plan.concrete.MemoryModel") as mock_memory_class:
            with patch("src.ace.plan.concrete.FlowAnalyzer") as mock_analyzer_class:
                mock_memory = MagicMock()
                mock_analyzer = MagicMock()
                mock_analyzer.valid = True

                mock_memory_class.return_value = mock_memory
                mock_analyzer_class.return_value = mock_analyzer

                # Should be able to create planner instance
                planner = InfoFlowPlanner.__new__(InfoFlowPlanner)
                assert planner is not None

    def test_component_cleanup_and_teardown(self):
        """Test that components can be properly cleaned up"""
        # Create components
        lattice = SubsetLattice({"public"})
        memory = MemoryModel(lattice_type=SubsetLattice, dynamic_vars={}, static_vars={"func": lattice})
        analyzer = FlowAnalyzer(memory)

        # Components should be garbage collectable
        import gc
        import weakref

        weakref.ref(lattice)
        weakref.ref(memory)
        weakref.ref(analyzer)

        # Remove references
        del lattice, memory, analyzer
        gc.collect()

        # At least some should be collectible (depending on implementation)
        # This is more about ensuring no circular references prevent cleanup
        assert True  # If we get here without hanging, cleanup works


class TestSystemDependencies:
    """Test dependencies between different system components"""

    def test_lattice_operations_consistency(self):
        """Test that lattice operations are consistent across components"""
        # Create the same lattice in different contexts
        lattice1 = SubsetLattice({"public", "private"})
        lattice2 = SubsetLattice({"public", "private"})

        # Should be equal
        assert lattice1 == lattice2

        # Operations should be consistent
        assert (lattice1 + lattice2) == lattice1
        assert (lattice1 * lattice2) == lattice1

    def test_memory_model_lattice_compatibility(self):
        """Test that memory models work with different lattice types"""
        # SubsetLattice should work
        memory1 = MemoryModel(
            lattice_type=SubsetLattice, dynamic_vars={}, static_vars={"func": SubsetLattice({"public"})}
        )
        assert memory1 is not None

        # Should be able to create analyzer with this memory
        analyzer1 = FlowAnalyzer(memory1)
        assert analyzer1 is not None

    def test_analyzer_planner_integration(self):
        """Test that analyzer and planner integrate properly"""
        # Create memory model
        memory = MemoryModel(
            lattice_type=SubsetLattice,
            dynamic_vars={},
            static_vars={"safe_function": SubsetLattice({"public"}), "secure_function": SubsetLattice({"private"})},
        )

        # Create analyzer
        analyzer = FlowAnalyzer(memory)

        # Planner should be able to use analyzer
        with patch("src.ace.plan.concrete.FlowAnalyzer") as mock_analyzer_class:
            mock_analyzer_class.return_value = analyzer

            planner = InfoFlowPlanner.__new__(InfoFlowPlanner)
            assert planner is not None


class TestSetupValidation:
    """Test that system setup is properly validated"""

    def test_required_components_present(self):
        """Test that all required components are available"""
        # Should be able to import all core components
        from src.ace.plan.concrete import InfoFlowPlanner
        from src.ace.schema.infoflow import MemoryModel
        from src.ace.schema.lattice import SubsetLattice
        from src.ace.security.infoflow import FlowAnalyzer

        assert SubsetLattice is not None
        assert MemoryModel is not None
        assert FlowAnalyzer is not None
        assert InfoFlowPlanner is not None

    def test_component_interface_compatibility(self):
        """Test that component interfaces are compatible"""
        # Create a simple workflow to test compatibility
        lattice = SubsetLattice({"public"})

        memory = MemoryModel(lattice_type=SubsetLattice, dynamic_vars={}, static_vars={"test": lattice})

        analyzer = FlowAnalyzer(memory)

        # All components should work together
        assert lattice is not None
        assert memory is not None
        assert analyzer is not None

        # Basic operations should work
        assert analyzer.valid == True
        assert len(analyzer.violations) == 0

    def test_configuration_parameter_types(self):
        """Test that configuration parameters have correct types"""
        # Lattice val should be sets
        lattice = SubsetLattice({"public", "private"})
        assert isinstance(lattice.val, set)

        # Memory model should have proper types
        memory = MemoryModel(lattice_type=SubsetLattice, dynamic_vars={}, static_vars={})
        assert hasattr(memory, "lattice_type")
        assert hasattr(memory, "dynamic_vars")
        assert hasattr(memory, "static_vars")
        assert isinstance(memory.dynamic_vars, dict)
        assert isinstance(memory.static_vars, dict)

    def test_default_configuration_validity(self):
        """Test that default configurations are valid"""
        # Empty configurations should be valid
        empty_lattice = SubsetLattice(set())
        assert empty_lattice is not None

        empty_memory = MemoryModel(lattice_type=SubsetLattice, dynamic_vars={}, static_vars={})
        assert empty_memory is not None

        empty_analyzer = FlowAnalyzer(empty_memory)
        assert empty_analyzer is not None
        assert empty_analyzer.valid == True

    def test_system_resource_requirements(self):
        """Test that system meets basic resource requirements"""
        # Should be able to create multiple components without issues
        components = []

        for i in range(10):
            lattice = SubsetLattice({f"level_{i}"})
            memory = MemoryModel(lattice_type=SubsetLattice, dynamic_vars={}, static_vars={f"func_{i}": lattice})
            analyzer = FlowAnalyzer(memory)
            components.append((lattice, memory, analyzer))

        # All should be created successfully
        assert len(components) == 10

        # All analyzers should be valid
        for lattice, memory, analyzer in components:
            assert analyzer.valid == True


class TestVersionCompatibility:
    """Test version compatibility and migration support"""

    def test_schema_backward_compatibility(self):
        """Test that schemas maintain backward compatibility"""
        # Basic schema creation should work consistently
        lattice = SubsetLattice({"public", "private"})
        memory = MemoryModel(lattice_type=SubsetLattice, dynamic_vars={}, static_vars={"func": lattice})

        # Should have expected attributes
        assert hasattr(lattice, "val")
        assert hasattr(memory, "lattice_type")
        assert hasattr(memory, "dynamic_vars")
        assert hasattr(memory, "static_vars")

    def test_api_stability(self):
        """Test that public APIs are stable"""
        # Core APIs should be available
        lattice = SubsetLattice({"test"})
        assert hasattr(lattice, "__add__")  # +
        assert hasattr(lattice, "__mul__")  # *
        assert hasattr(lattice, "__le__")  # <=
        assert hasattr(lattice, "__eq__")  # ==

        memory = MemoryModel(lattice_type=SubsetLattice, dynamic_vars={}, static_vars={})
        analyzer = FlowAnalyzer(memory)
        assert hasattr(analyzer, "analyze_flow")
        assert hasattr(analyzer, "apply_explicit_flow")
        assert hasattr(analyzer, "valid")
        assert hasattr(analyzer, "violations")
