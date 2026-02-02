"""Tests for ESN core functionality."""

import numpy as np
import pytest
import tempfile
import os

from rc import ESN, ESNConfig, StandardDynamics, LeakyDynamics, ES2NDynamics


class TestESNConfig:
    """Tests for ESNConfig validation."""
    
    def test_valid_config(self):
        """Test creating a valid config."""
        config = ESNConfig(N=100, input_dim=3)
        assert config.N == 100
        assert config.input_dim == 3
        assert config.spectral_radius == 0.9
        assert config.mode == "standard"
    
    def test_invalid_N(self):
        """Test that invalid N raises ValueError."""
        with pytest.raises(ValueError, match="N must be a positive integer"):
            ESNConfig(N=0, input_dim=3)
        with pytest.raises(ValueError, match="N must be a positive integer"):
            ESNConfig(N=-5, input_dim=3)
    
    def test_invalid_input_dim(self):
        """Test that invalid input_dim raises ValueError."""
        with pytest.raises(ValueError, match="input_dim must be a positive integer"):
            ESNConfig(N=100, input_dim=0)
    
    def test_invalid_spectral_radius(self):
        """Test that invalid spectral_radius raises ValueError."""
        with pytest.raises(ValueError, match="spectral_radius must be positive"):
            ESNConfig(N=100, input_dim=3, spectral_radius=-0.5)
    
    def test_invalid_sparsity(self):
        """Test that invalid sparsity raises ValueError."""
        with pytest.raises(ValueError, match="sparsity must be in"):
            ESNConfig(N=100, input_dim=3, sparsity=1.5)
        with pytest.raises(ValueError, match="sparsity must be in"):
            ESNConfig(N=100, input_dim=3, sparsity=-0.1)
    
    def test_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be one of"):
            ESNConfig(N=100, input_dim=3, mode="invalid_mode")


class TestESNCreation:
    """Tests for ESN instantiation."""
    
    def test_create_with_config(self):
        """Test creating ESN with config object."""
        config = ESNConfig(N=50, input_dim=3, seed=42)
        esn = ESN(config)
        assert esn.N == 50
        assert esn.input_dim == 3
        assert not esn.is_trained
    
    def test_create_with_params(self):
        """Test creating ESN with individual parameters."""
        esn = ESN(N=50, input_dim=3, spectral_radius=0.95, seed=42)
        assert esn.N == 50
        assert esn.input_dim == 3
        assert esn.config.spectral_radius == 0.95
    
    def test_create_conflict_raises(self):
        """Test that providing both config and params raises error."""
        config = ESNConfig(N=50, input_dim=3)
        with pytest.raises(ValueError, match="Cannot specify both config and individual parameters"):
            ESN(config, N=100)
    
    def test_reservoir_shapes(self):
        """Test that reservoir matrices have correct shapes."""
        esn = ESN(N=50, input_dim=3, seed=42)
        assert esn.Wr.shape == (50, 50)
        assert esn.Wx.shape == (50, 3)
        assert esn.b.shape == (50,)
        assert esn.r.shape == (50,)


class TestESNTrain:
    """Tests for ESN training."""
    
    def test_train_sets_output_weights(self):
        """Test that training sets Wout and Wout_bias."""
        esn = ESN(N=50, input_dim=3, seed=42)
        data = np.random.randn(3, 1000)
        esn.train(data, washout=100)
        assert esn.is_trained
        assert esn.Wout is not None
        assert esn.Wout_bias is not None
        assert esn.Wout.shape == (3, 50)
        assert esn.Wout_bias.shape == (3,)
    
    def test_train_invalid_data_shape(self):
        """Test that train raises on invalid data shape."""
        esn = ESN(N=50, input_dim=3, seed=42)
        with pytest.raises(ValueError, match="x_train must be 2D"):
            esn.train(np.random.randn(3000), washout=100)
    
    def test_train_wrong_input_dim(self):
        """Test that train raises when data has wrong input_dim."""
        esn = ESN(N=50, input_dim=3, seed=42)
        with pytest.raises(ValueError, match="first dimension must match input_dim"):
            esn.train(np.random.randn(5, 1000), washout=100)


class TestESNPredict:
    """Tests for ESN prediction."""
    
    @pytest.fixture
    def trained_esn(self):
        """Create and train an ESN for testing."""
        esn = ESN(N=50, input_dim=3, seed=42)
        data = np.random.randn(3, 1000)
        esn.train(data, washout=100)
        return esn
    
    def test_predict_requires_training(self):
        """Test that predict raises if not trained."""
        esn = ESN(N=50, input_dim=3, seed=42)
        warmup = np.random.randn(3, 50)
        with pytest.raises(RuntimeError, match="must be trained"):
            esn.predict(warmup, steps=100)
    
    def test_predict_returns_correct_shapes(self, trained_esn):
        """Test that predict returns correct shapes."""
        warmup = np.random.randn(3, 50)
        predictions, states = trained_esn.predict(warmup, steps=100)
        assert predictions.shape == (3, 100)
        assert states.shape == (50, 100)


class TestESNSaveLoad:
    """Tests for ESN save/load functionality."""
    
    def test_save_load_roundtrip(self):
        """Test that save/load preserves ESN state."""
        esn = ESN(N=50, input_dim=3, seed=42)
        data = np.random.randn(3, 1000)
        esn.train(data, washout=100)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.npz")
            esn.save(path)
            
            loaded = ESN.load(path)
            
            assert loaded.N == esn.N
            assert loaded.input_dim == esn.input_dim
            assert np.allclose(loaded.Wout, esn.Wout)
            assert np.allclose(loaded.Wout_bias, esn.Wout_bias)


class TestESNDynamics:
    """Tests for different ESN dynamics modes."""
    
    def test_standard_dynamics(self):
        """Test ESN with standard dynamics."""
        esn = ESN(N=50, input_dim=3, mode="standard", seed=42)
        assert isinstance(esn.dynamics, StandardDynamics)
    
    def test_leaky_dynamics(self):
        """Test ESN with leaky dynamics."""
        esn = ESN(N=50, input_dim=3, mode="leaky", leaky_rate=0.3, seed=42)
        assert isinstance(esn.dynamics, LeakyDynamics)
    
    def test_es2n_dynamics(self):
        """Test ESN with ES2N dynamics."""
        esn = ESN(N=50, input_dim=3, mode="es2n", beta=0.5, seed=42)
        assert isinstance(esn.dynamics, ES2NDynamics)
