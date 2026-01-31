def choose_vispy_backend():
    """Try to set a VisPy GUI backend in order of preference."""
    import vispy
    backend_set = False

    try:
        import PyQt5
        vispy.use('pyqt5', 'force')
        backend_set = True
    except ImportError:
        try:
            import PySide6
            vispy.use('pyside6', 'force')
            backend_set = True
        except ImportError:
            try:
                import glfw
                vispy.use('glfw', 'force')
                backend_set = True
            except ImportError:
                backend_set = False

    if not backend_set:
        print(
            "WARNING: No VisPy GUI backend found. "
            "Install pyqt5, pyside6, or glfw to visualize."
        )
    return backend_set
