import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use('TkAgg')  # Force TkAgg backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from dsp_signal.signal import Signal
from dsp_signal.file_io import ReadSignalFile
from dsp_signal.operations import derivative_signal, convolve_signals, moving_average_signal
from dsp_signal.fourier import fourier_transform_signal, inverse_fourier_transform
from utils.comparison import compare_signals
from gui.styles import setup_styles

class DSPApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("DSP Signal Processing Tool")

        try:
            self.root.state('zoomed')
        except Exception:
            try:
                self.root.attributes('-zoomed', True)
            except Exception:
                pass

        # Store loaded signals
        self.signals = []
        self.result_signal = None

        # create styles for colored buttons
        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use('clam')
        except Exception:
            pass

        # Setup styles
        self.style = ttk.Style(self.root)
        setup_styles(self.style)
        
        # Add menu bar for signal generation
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        
        signal_gen_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='Signal Generation', menu=signal_gen_menu)
        signal_gen_menu.add_command(label='Sine Wave', command=lambda: self.open_generate_dialog('sine'))
        signal_gen_menu.add_command(label='Cosine Wave', command=lambda: self.open_generate_dialog('cosine'))

        # build UI
        self.setup_gui()


    def setup_gui(self):
        # plot on left, controls on right
        plot_frame = ttk.Frame(self.root, padding="8")
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ---- Scrollable Control Panel ----
        control_container = ttk.Frame(self.root)
        control_container.pack(side=tk.RIGHT, fill=tk.Y)

        canvas = tk.Canvas(control_container, width=400)
        scrollbar = ttk.Scrollbar(control_container, orient="vertical", command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)

        control_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=control_frame, anchor="nw")

        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        control_frame.bind("<Configure>", _on_frame_configure)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Signal management section
        mgmt_frame = ttk.LabelFrame(control_frame, text="Signal Management", padding="6")
        mgmt_frame.pack(fill=tk.X, pady=(6, 8), padx=6)

        ttk.Button(mgmt_frame, text="Load Signal", style='Load.TButton',
                   command=self.load_signal).pack(fill=tk.X, pady=6)
        ttk.Button(mgmt_frame, text="Clear All", style='ClearAll.TButton',
                   command=self.clear_all).pack(fill=tk.X, pady=6)

        self.signals_listbox = tk.Listbox(mgmt_frame, height=8, selectmode=tk.MULTIPLE,
                                          bg="#fafafa", fg="#111", font=('Consolas', 10))
        self.signals_listbox.pack(fill=tk.X, pady=6)

        # Basic Operations
        ops_frame = ttk.LabelFrame(control_frame, text="Basic Operations", padding="6")
        ops_frame.pack(fill=tk.X, pady=(6, 8), padx=6)

        ttk.Button(ops_frame, text="Add Signals", style='Add.TButton',
                   command=self.add_signals).pack(fill=tk.X, pady=6)
        ttk.Button(ops_frame, text="Subtract Signals", style='Subtract.TButton',
                   command=self.subtract_signals).pack(fill=tk.X, pady=6)

        const_frame = ttk.Frame(ops_frame)
        const_frame.pack(fill=tk.X, pady=6, padx=4)
        ttk.Label(const_frame, text="Constant:").pack(side=tk.LEFT)
        self.const_entry = ttk.Entry(const_frame, width=8)
        self.const_entry.pack(side=tk.LEFT, padx=6)
        self.const_entry.insert(0, "1.0")
        ttk.Button(const_frame, text="Multiply", style='Multiply.TButton',
                   command=self.multiply_constant).pack(side=tk.LEFT)

        shift_frame = ttk.Frame(ops_frame)
        shift_frame.pack(fill=tk.X, pady=6, padx=4)
        ttk.Label(shift_frame, text="Shift (k):").pack(side=tk.LEFT)
        self.shift_entry = ttk.Entry(shift_frame, width=8)
        self.shift_entry.pack(side=tk.LEFT, padx=6)
        self.shift_entry.insert(0, "0")
        ttk.Button(shift_frame, text="Shift", style='Shift.TButton',
                   command=self.shift_signal).pack(side=tk.LEFT)

        ttk.Button(ops_frame, text="Fold Signal", style='Fold.TButton',
                   command=self.fold_signal).pack(fill=tk.X, pady=6)
        ttk.Button(ops_frame, text="Quantize Signal", style='Multiply.TButton',
                   command=self.quantize_signal).pack(fill=tk.X, pady=6)

        dsp_frame = ttk.LabelFrame(control_frame, text="Advanced DSP Operations", padding="6")
        dsp_frame.pack(fill=tk.X, pady=(6, 8), padx=6)

        # Derivative
        deriv_frame = ttk.Frame(dsp_frame)
        deriv_frame.pack(fill=tk.X, pady=6, padx=4)
        ttk.Label(deriv_frame, text="Derivative Level:").pack(side=tk.LEFT)
        self.deriv_level = ttk.Combobox(deriv_frame, width=5, values=["1", "2"], state="readonly")
        self.deriv_level.pack(side=tk.LEFT, padx=6)
        self.deriv_level.set("1")
        ttk.Button(deriv_frame, text="Derivative", style='Derivative.TButton',
                   command=self.calculate_derivative).pack(side=tk.LEFT)

        # Convolution
        ttk.Button(dsp_frame, text="Convolution", style='Convolution.TButton',
                   command=self.calculate_convolution).pack(fill=tk.X, pady=6)

        # Moving Average
        mov_avg_frame = ttk.Frame(dsp_frame)
        mov_avg_frame.pack(fill=tk.X, pady=6, padx=4)
        ttk.Label(mov_avg_frame, text="Window Size:").pack(side=tk.LEFT)
        self.window_size_entry = ttk.Entry(mov_avg_frame, width=8)
        self.window_size_entry.pack(side=tk.LEFT, padx=6)
        self.window_size_entry.insert(0, "3")
        ttk.Button(mov_avg_frame, text="Moving Average", style='MovingAvg.TButton',
                   command=self.calculate_moving_average).pack(side=tk.LEFT)

        # Comparison
        compare_frame = ttk.LabelFrame(control_frame, text="Signal Comparison", padding="6")
        compare_frame.pack(fill=tk.X, pady=(6, 8), padx=6)
        ttk.Button(compare_frame, text="Compare with File", style='Compare.TButton',
                   command=self.compare_with_file).pack(fill=tk.X, pady=6)
        
        # Fourier Transform Operations
        fourier_frame = ttk.LabelFrame(control_frame, text="Fourier Transform", padding="6")
        fourier_frame.pack(fill=tk.X, pady=(6, 8), padx=6)

        # Sampling frequency input
        fs_frame = ttk.Frame(fourier_frame)
        fs_frame.pack(fill=tk.X, pady=6, padx=4)
        ttk.Label(fs_frame, text="Sampling Freq (Hz):").pack(side=tk.LEFT)
        self.fs_fourier_entry = ttk.Entry(fs_frame, width=10)
        self.fs_fourier_entry.pack(side=tk.LEFT, padx=6)
        self.fs_fourier_entry.insert(0, "100.0")

        # DFT Button
        ttk.Button(fourier_frame, text="Compute DFT", style='Derivative.TButton',
                command=self.compute_dft).pack(fill=tk.X, pady=6)

        # IDFT Button
        ttk.Button(fourier_frame, text="Reconstruct (IDFT)", style='Convolution.TButton',
                command=self.compute_idft).pack(fill=tk.X, pady=6)

        # Store Fourier results
        self.magnitude_signal = None
        self.phase_signal = None

        # Display options
        disp_frame = ttk.LabelFrame(control_frame, text="Display Options", padding=6)
        disp_frame.pack(fill=tk.X, pady=(6, 8), padx=6)

        self.display_mode = tk.StringVar(value='discrete')
        self.fs_for_time_plot = tk.DoubleVar(value=100.0)

        ttk.Radiobutton(disp_frame, text="Discrete", variable=self.display_mode,
                        value='discrete').pack(anchor=tk.W)
        ttk.Radiobutton(disp_frame, text="Continuous", variable=self.display_mode,
                        value='continuous').pack(anchor=tk.W)

        fs_frame = ttk.Frame(disp_frame)
        fs_frame.pack(fill=tk.X, pady=6)
        ttk.Label(fs_frame, text="Time-plot fs (Hz):").pack(side=tk.LEFT)
        self.fs_entry = ttk.Entry(fs_frame, width=8, textvariable=self.fs_for_time_plot)
        self.fs_entry.pack(side=tk.LEFT, padx=6)

        # Plot buttons
        plot_btn_frame = ttk.LabelFrame(control_frame, text="Plotting", padding="6")
        plot_btn_frame.pack(fill=tk.X, pady=(6, 8), padx=6)
        ttk.Button(plot_btn_frame, text="Plot Selected", style='Plot.TButton',
                   command=self.plot_selected).pack(fill=tk.X, pady=6)
        ttk.Button(plot_btn_frame, text="Plot Two Signals", style='Plot.TButton',
                   command=self.plot_two_selected).pack(fill=tk.X, pady=6)

        # Results section
        result_frame = ttk.LabelFrame(control_frame, text="Results", padding="6")
        result_frame.pack(fill=tk.X, pady=(6, 8), padx=6)
        ttk.Button(result_frame, text="Save Result", style='Save.TButton',
                   command=self.save_result).pack(fill=tk.X, pady=6)
        ttk.Button(result_frame, text="Clear Result", style='ClearResult.TButton',
                   command=self.clear_result).pack(fill=tk.X, pady=6)

        self.setup_plot_area(plot_frame)

    def setup_plot_area(self, parent):
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('Index (n)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Signal Visualization')    

    def calculate_derivative(self):
        """Calculate derivative of selected signal"""
        selected_signals = self.get_selected_signals()
        if len(selected_signals) != 1:
            messagebox.showwarning("Warning", "Please select exactly one signal")
            return

        try:
            derivative_level = int(self.deriv_level.get())
            signal = selected_signals[0]
            signal_dict = signal.to_dict()

            result_dict = derivative_signal(signal_dict, derivative_level)
            self.result_signal = Signal.from_dict(result_dict, f"Derivative_{derivative_level}_{signal.name}")

            self.plot_result()
            messagebox.showinfo("Success", f"Derivative (level {derivative_level}) calculated successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Derivative calculation failed: {str(e)}")

    def calculate_convolution(self):
        """Calculate convolution of two selected signals"""
        selected_signals = self.get_selected_signals()
        if len(selected_signals) != 2:
            messagebox.showwarning("Warning", "Please select exactly two signals for convolution")
            return

        try:
            signal1 = selected_signals[0]
            signal2 = selected_signals[1]

            signal1_dict = signal1.to_dict()
            signal2_dict = signal2.to_dict()

            result_dict = convolve_signals(signal1_dict, signal2_dict)
            self.result_signal = Signal.from_dict(result_dict, f"Convolution_{signal1.name}_{signal2.name}")

            self.plot_result()
            messagebox.showinfo("Success", "Convolution calculated successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Convolution failed: {str(e)}")

    def calculate_moving_average(self):
        """Calculate moving average of selected signal"""
        selected_signals = self.get_selected_signals()
        if len(selected_signals) != 1:
            messagebox.showwarning("Warning", "Please select exactly one signal")
            return

        try:
            window_size = int(self.window_size_entry.get())
            signal = selected_signals[0]
            signal_dict = signal.to_dict()

            result_dict = moving_average_signal(signal_dict, window_size)
            self.result_signal = Signal.from_dict(result_dict, f"MovingAvg_{window_size}_{signal.name}")

            self.plot_result()
            messagebox.showinfo("Success", f"Moving average (window={window_size}) calculated successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Moving average calculation failed: {str(e)}")

    def compare_with_file(self):
        """Compare result signal with expected signal from file"""
        if self.result_signal is None:
            messagebox.showwarning("Warning", "No result signal to compare")
            return

        filename = filedialog.askopenfilename(
            title="Select Expected Signal File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if not filename:
            return

        try:
            success = compare_signals(
                self.result_signal.indices,
                self.result_signal.samples,
                filename
            )

            if success:
                messagebox.showinfo("Comparison Result", "Test case passed successfully!")
            else:
                messagebox.showwarning("Comparison Result", "Test case failed! Check console for details.")

        except Exception as e:
            messagebox.showerror("Error", f"Comparison failed: {str(e)}")

    def read_signal_file(self, filename):
        # Using the same ReadSignalFile function for consistency
        return ReadSignalFile(filename)

    def load_signal(self):
        filename = filedialog.askopenfilename(
            title="Select Signal File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not filename:
            return
        try:
            indices, samples = self.read_signal_file(filename)
            signal_name = filename.split("/")[-1]
            signal = Signal(indices, samples, signal_name)
            self.signals.append(signal)
            self.signals_listbox.insert(tk.END, f"{signal_name} ({len(indices)} samples)")
            messagebox.showinfo("Success", f"Signal loaded successfully!\n{len(indices)} samples")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load signal: {str(e)}")

    def get_selected_signals(self):
        selected_indices = self.signals_listbox.curselection()
        return [self.signals[i] for i in selected_indices]

    def add_signals(self):
        selected_signals = self.get_selected_signals()
        if len(selected_signals) < 2:
            messagebox.showwarning("Warning", "Please select at least 2 signals to add")
            return
        try:
            all_indices = set()
            for signal in selected_signals:
                all_indices.update(signal.indices)
            common_indices = sorted(all_indices)
            result_samples = [0.0] * len(common_indices)
            for i, idx in enumerate(common_indices):
                for signal in selected_signals:
                    if idx in signal.indices:
                        pos = signal.indices.index(idx)
                        result_samples[i] += signal.samples[pos]
            self.result_signal = Signal(common_indices, result_samples, "Addition Result")
            self.plot_result()
            messagebox.showinfo("Success", "Signals added successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Addition failed: {str(e)}")

    def subtract_signals(self):
        selected_signals = self.get_selected_signals()
        if len(selected_signals) < 2:
            messagebox.showwarning("Warning", "Please select at least 2 signals to subtract")
            return
        try:
            all_indices = set()
            for signal in selected_signals:
                all_indices.update(signal.indices)
            common_indices = sorted(all_indices)
            result_samples = [0.0] * len(common_indices)
            for i, idx in enumerate(common_indices):
                if idx in selected_signals[0].indices:
                    pos = selected_signals[0].indices.index(idx)
                    result_samples[i] += selected_signals[0].samples[pos]
                for signal in selected_signals[1:]:
                    if idx in signal.indices:
                        pos = signal.indices.index(idx)
                        result_samples[i] -= signal.samples[pos]
            self.result_signal = Signal(common_indices, result_samples, "Subtraction Result")
            self.plot_result()
            messagebox.showinfo("Success", "Signals subtracted successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Subtraction failed: {str(e)}")

    def multiply_constant(self):
        selected_signals = self.get_selected_signals()
        if len(selected_signals) != 1:
            messagebox.showwarning("Warning", "Please select exactly one signal")
            return
        try:
            constant = float(self.const_entry.get())
            signal = selected_signals[0]
            result_samples = [s * constant for s in signal.samples]
            self.result_signal = Signal(signal.indices.copy(), result_samples, f"Multiplied by {constant}")
            self.plot_result()
            messagebox.showinfo("Success", f"Signal multiplied by {constant} successfully!")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for constant")
        except Exception as e:
            messagebox.showerror("Error", f"Multiplication failed: {str(e)}")

    def shift_signal(self):
        selected_signals = self.get_selected_signals()
        if len(selected_signals) != 1:
            messagebox.showwarning("Warning", "Please select exactly one signal")
            return
        try:
            k = int(self.shift_entry.get())
            signal = selected_signals[0]
            new_indices = [idx - k for idx in signal.indices]
            self.result_signal = Signal(new_indices, signal.samples.copy(), f"Shifted by {k}")
            self.plot_result()
            messagebox.showinfo("Success", f"Signal shifted by {k} successfully!")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer for shift")
        except Exception as e:
            messagebox.showerror("Error", f"Shifting failed: {str(e)}")

    def fold_signal(self):
        selected_signals = self.get_selected_signals()
        if len(selected_signals) != 1:
            messagebox.showwarning("Warning", "Please select exactly one signal")
            return
        try:
            signal = selected_signals[0]
            new_indices = [-idx for idx in reversed(signal.indices)]
            new_samples = list(reversed(signal.samples))
            self.result_signal = Signal(new_indices, new_samples, "Folded Signal")
            self.plot_result()
            messagebox.showinfo("Success", "Signal folded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Folding failed: {str(e)}")

    def quantize_signal(self):
        selected_signals = self.get_selected_signals()
        if len(selected_signals) != 1:
            messagebox.showwarning("Warning", "Please select exactly one signal to quantize")
            return
    
    def compute_dft(self):
        """Compute Discrete Fourier Transform of selected signal"""
        selected_signals = self.get_selected_signals()
        if len(selected_signals) != 1:
            messagebox.showwarning("Warning", "Please select exactly one signal for DFT")
            return
        
        try:
            # Get sampling frequency
            sampling_freq = float(self.fs_fourier_entry.get())
            if sampling_freq <= 0:
                raise ValueError("Sampling frequency must be positive")
            
            signal = selected_signals[0]
            
            # Compute Fourier Transform
            self.magnitude_signal, self.phase_signal = fourier_transform_signal(signal, sampling_freq)
            
            # Plot magnitude and phase
            self.plot_fourier_results()
            
            messagebox.showinfo("Success", 
                              f"DFT computed successfully!\n"
                              f"Sampling Frequency: {sampling_freq} Hz\n"
                              f"Nyquist Frequency: {sampling_freq/2:.2f} Hz")
            
        except ValueError as ve:
            messagebox.showerror("Error", f"Invalid input: {str(ve)}")
        except Exception as e:
            messagebox.showerror("Error", f"DFT computation failed: {str(e)}")

    def compute_idft(self):
        """Reconstruct signal using Inverse DFT"""
        if self.magnitude_signal is None:
            messagebox.showwarning("Warning", "No Fourier transform results available.\nPlease compute DFT first.")
            return
        
        try:
            # Reconstruct signal from magnitude (and phase if available)
            reconstructed_signal = inverse_fourier_transform(
                self.magnitude_signal, 
                self.phase_signal
            )
            
            self.result_signal = reconstructed_signal
            self.plot_result()
            
            # Compare with original if available
            selected_signals = self.get_selected_signals()
            if selected_signals:
                original = selected_signals[0]
                # Calculate reconstruction error
                if len(original.samples) == len(reconstructed_signal.samples):
                    error = np.sqrt(np.mean((
                        np.array(original.samples[:len(reconstructed_signal.samples)]) - 
                        np.array(reconstructed_signal.samples)
                    )**2))
                    messagebox.showinfo("Success", 
                                      f"Signal reconstructed successfully!\n"
                                      f"Reconstruction RMSE: {error:.6f}")
                else:
                    messagebox.showinfo("Success", "Signal reconstructed successfully!")
            else:
                messagebox.showinfo("Success", "Signal reconstructed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"IDFT computation failed: {str(e)}")

    def plot_fourier_results(self):
        """Plot magnitude and phase spectra"""
        if self.magnitude_signal is None:
            return
        
        # Create a new figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot magnitude spectrum
        indices_mag, samples_mag = self.magnitude_signal.get_plot_data()
        ax1.stem(indices_mag, samples_mag, linefmt='b-', markerfmt='bo', basefmt=' ')
        ax1.set_xlabel('Frequency Bin (k)')
        ax1.set_ylabel('Magnitude')
        ax1.set_title(f'Magnitude Spectrum - {self.magnitude_signal.name}')
        ax1.grid(True, alpha=0.3)
        
        # Plot phase spectrum if available
        if self.phase_signal is not None:
            indices_phase, samples_phase = self.phase_signal.get_plot_data()
            ax2.stem(indices_phase, samples_phase, linefmt='r-', markerfmt='ro', basefmt=' ')
            ax2.set_xlabel('Frequency Bin (k)')
            ax2.set_ylabel('Phase (degrees)')
            ax2.set_title(f'Phase Spectrum - {self.phase_signal.name}')
        else:
            ax2.text(0.5, 0.5, 'No phase information available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes)
            ax2.set_title('Phase Spectrum')
        
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Show in a separate window
        plt.show()

    def plot_fourier_single(self):
        """Plot Fourier results in the main canvas"""
        if self.magnitude_signal is None:
            return
        
        self.ax.clear()
        
        # Plot magnitude
        indices, samples = self.magnitude_signal.get_plot_data()
        self.ax.stem(indices, samples, linefmt='b-', markerfmt='bo', basefmt=' ', 
                    label='Magnitude')
        
        # Plot phase on secondary axis if available
        if self.phase_signal is not None:
            ax2 = self.ax.twinx()
            indices_phase, samples_phase = self.phase_signal.get_plot_data()
            ax2.plot(indices_phase, samples_phase, 'r--', label='Phase (degrees)', alpha=0.7)
            ax2.set_ylabel('Phase (degrees)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Combine legends
            lines1, labels1 = self.ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            self.ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            self.ax.legend()
        
        self.ax.set_xlabel('Frequency Bin (k)')
        self.ax.set_ylabel('Magnitude', color='b')
        self.ax.tick_params(axis='y', labelcolor='b')
        self.ax.set_title('Fourier Transform Results')
        self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()

    def plot_selected(self):
        selected_signals = self.get_selected_signals()
        if not selected_signals:
            messagebox.showwarning("Warning", "Please select at least one signal to plot")
            return
        self.ax.clear()
        mode = self.display_mode.get()
        fs_time = float(self.fs_entry.get()) if self.fs_entry.get() else float(self.fs_for_time_plot.get())

        for signal in selected_signals:
            if mode == 'discrete':
                indices, samples = signal.get_plot_data()
                if indices and samples:
                    self.ax.stem(indices, samples, linefmt='-', markerfmt='o', basefmt=' ', label=signal.name)
            else:
                t_samples, y_samples = signal.get_time_series(fs_time)
                if t_samples.size == 0:
                    continue
                t_dense = np.linspace(t_samples.min(), t_samples.max(), max(200, len(t_samples) * 10))
                y_dense = np.interp(t_dense, t_samples, y_samples)
                self.ax.plot(t_dense, y_dense, '-', label=signal.name)

        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('n (discrete) or t (s)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Signal Visualization')
        self.ax.legend()
        self.canvas.draw()

    def plot_two_selected(self):
        selected_signals = self.get_selected_signals()
        if len(selected_signals) < 2:
            messagebox.showwarning("Warning", "Please select at least two signals to plot together")
            return
        self.ax.clear()
        mode = self.display_mode.get()
        fs_time = float(self.fs_entry.get()) if self.fs_entry.get() else float(self.fs_for_time_plot.get())

        for signal in selected_signals[:2]:
            if mode == 'discrete':
                indices, samples = signal.get_plot_data()
                if indices and samples:
                    self.ax.stem(indices, samples, linefmt='-', markerfmt='o', basefmt=' ', label=signal.name)
            else:
                t_samples, y_samples = signal.get_time_series(fs_time)
                if t_samples.size == 0:
                    continue
                t_dense = np.linspace(t_samples.min(), t_samples.max(), max(200, len(t_samples) * 10))
                y_dense = np.interp(t_dense, t_samples, y_samples)
                self.ax.plot(t_dense, y_dense, '-', label=signal.name)

        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('n (discrete) or t (s)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Two Signals')
        self.ax.legend()
        self.canvas.draw()

    def plot_result(self):
        if self.result_signal is None:
            return
        self.ax.clear()
        indices, samples = self.result_signal.get_plot_data()
        self.ax.stem(indices, samples, linefmt='r-', markerfmt='ro', basefmt=' ', label=self.result_signal.name)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('Index (n)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Operation Result')
        self.ax.legend()
        self.canvas.draw()

    def save_result(self):
        if self.result_signal is None:
            messagebox.showwarning("Warning", "No result to save")
            return
        filename = filedialog.asksaveasfilename(
            title="Save Result Signal",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not filename:
            return
        try:
            with open(filename, 'w') as f:
                f.write("0\n")
                f.write("0\n")
                f.write(f"{len(self.result_signal.indices)}\n")
                for idx, sample in zip(self.result_signal.indices, self.result_signal.samples):
                    f.write(f"{idx} {sample}\n")
            messagebox.showinfo("Success", f"Result saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")

    def clear_result(self):
        self.result_signal = None
        self.ax.clear()
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('Index (n)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Signal Visualization')
        self.canvas.draw()

    def clear_all(self):
        self.signals.clear()
        self.result_signal = None
        self.signals_listbox.delete(0, tk.END)
        self.clear_result()
        messagebox.showinfo("Info", "All signals cleared")

