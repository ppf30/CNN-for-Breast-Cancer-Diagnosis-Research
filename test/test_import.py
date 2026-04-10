import unittest

class TestImport(unittest.TestCase):
    """
    Pruebas unitarias para verificar la visibilidad de las clases dentro del módulo `utils`.
    """

    def test_downsample_is_accessible(self):
        """Verifica que `DownSample` es accesible desde `utils`."""
        from unet.utils import DownSample
        self.assertTrue(hasattr(DownSample, "__init__"))

    def test_upsample_is_accessible(self):
        """Verifica que `UpSample` es accesible desde `utils`."""
        from unet.utils import UpSample
        self.assertTrue(hasattr(UpSample, "__init__"))

    def test_upsample_atteention_is_accessible(self):
            """Verifica que `UpSampleAttention` es accesible desde `utils`."""
            from unet.utils import UpSampleAttention
            self.assertTrue(hasattr(UpSampleAttention, "__init__"))


    
    def test_doubleconv_is_not_accessible(self):
        """Comprueba que `DoubleConv` NO se puede importar desde `utils`."""
        try:
            from unet.utils import DoubleConv  # Intentamos importar DoubleConv
            self.fail("Se pudo importar `DoubleConv`, pero no debería estar accesible.")  # Falla si no lanza ImportError
        except ImportError:
            pass  # El error esperado ocurrió, el test pasa

    
    
    def test_attentiongate_is_not_accessible(self):
        """Comprueba que `AttentionGate` NO se puede importar desde `utils`."""
        try:
            from unet.utils import AttentionGate  # Intentamos importar AttentionGate
            self.fail("Se pudo importar `AttentionGate`, pero no debería estar accesible.")  # Falla si no lanza ImportError
        except ImportError:
            pass  # El error esperado ocurrió, el test pasa

if __name__ == "__main__":
    unittest.main()
