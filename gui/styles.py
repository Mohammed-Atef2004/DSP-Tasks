def setup_styles(style):
    """Setup button styles for different operations"""
    try:
        style.theme_use('clam')
    except Exception:
        pass

    styles_config = {
        'Load.TButton': ('#1e88e5', '#1565c0'),
        'ClearAll.TButton': ('#e53935', '#b71c1c'),
        'Add.TButton': ('#43a047', '#2e7d32'),
        'Subtract.TButton': ('#fb8c00', '#ef6c00'),
        'Multiply.TButton': ('#8e24aa', '#6a1b9a'),
        'Shift.TButton': ('#00acc1', '#00838f'),
        'Fold.TButton': ('#6d6e71', '#424242'),
        'Plot.TButton': ('#1565c0', '#0d47a1'),
        'Save.TButton': ('#7cb342', '#558b2f'),
        'ClearResult.TButton': ('#ef5350', '#e53935'),
        'Derivative.TButton': ('#5e35b1', '#4527a0'),
        'Convolution.TButton': ('#00897b', '#00695c'),
        'MovingAvg.TButton': ('#f57c00', '#e65100'),
        'Compare.TButton': ('#546e7a', '#37474f'),
        'dft.TButton': ('#8e44ad', '#732d91'),
        'idft.TButton': ('#16a085', '#138d75'),
        'dcorr.TButton': ('#43a047', '#2e7d32'),
        'est.TButton': ('#fb8c00', '#ef6c00'),
        'class.TButton': ('#00897b', '#00695c'),
        'Filter.TButton': ('#2980b9', '#1a5276'),
    }

    for style_name, (bg_color, active_bg) in styles_config.items():
        style.configure(style_name, foreground='white', background=bg_color,
                        font=('Segoe UI', 10, 'bold'), padding=6)
        style.map(style_name, background=[('active', active_bg)])