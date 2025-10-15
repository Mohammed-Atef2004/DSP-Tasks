#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from typing import List, Tuple


class Signal:
    def __init__(self, indices=None, samples=None, name=""):
        self.indices = indices if indices is not None else []
        self.samples = samples if samples is not None else []
        self.name = name

    def __str__(self):
        return f"Signal '{self.name}': {len(self.indices)} samples"

    def get_plot_data(self):
        """Return data suitable for matplotlib plotting"""
        return self.indices, self.samples

    def get_time_series(self, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return time vector and samples sampled at rate fs for continuous plotting convenience."""
        if not self.indices:
            return np.array([]), np.array([])
        n = np.array(self.indices)
        t = n / fs
        y = np.array(self.samples)
        return t, y

class DSPApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("DSP Signal Processing Tool")
        try:
            # maximize window (cross-platform friendly)
            self.root.state('zoomed')
        except Exception:
            try:
                self.root.attributes('-zoomed', True)
            except Exception:
                pass

        # Store loaded signals
        self.signals = []
        self.result_signal = None

        # create styles for colored buttons (modern look)
        self.style = ttk.Style(self.root)
        # Ensure using a theme that supports styling well
        try:
            self.style.theme_use('clam')
        except Exception:
            pass

        # Add menu bar for signal generation
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        signal_gen_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='Signal Generation', menu=signal_gen_menu)
        signal_gen_menu.add_command(label='Sine Wave', command=lambda: self.open_generate_dialog('sine'))
        signal_gen_menu.add_command(label='Cosine Wave', command=lambda: self.open_generate_dialog('cosine'))

        # Define color styles - each operation gets its own TButton style
        self.style.configure('Load.TButton', foreground='white', background='#1e88e5', font=('Segoe UI', 10, 'bold'), padding=6)
        self.style.map('Load.TButton', background=[('active', '#1565c0')])

        self.style.configure('ClearAll.TButton', foreground='white', background='#e53935', font=('Segoe UI', 10, 'bold'), padding=6)
        self.style.map('ClearAll.TButton', background=[('active', '#b71c1c')])

        self.style.configure('Add.TButton', foreground='white', background='#43a047', font=('Segoe UI', 10, 'bold'), padding=6)
        self.style.map('Add.TButton', background=[('active', '#2e7d32')])

        self.style.configure('Subtract.TButton', foreground='white', background='#fb8c00', font=('Segoe UI', 10, 'bold'), padding=6)
        self.style.map('Subtract.TButton', background=[('active', '#ef6c00')])

        self.style.configure('Multiply.TButton', foreground='white', background='#8e24aa', font=('Segoe UI', 10, 'bold'), padding=6)
        self.style.map('Multiply.TButton', background=[('active', '#6a1b9a')])

        self.style.configure('Shift.TButton', foreground='white', background='#00acc1', font=('Segoe UI', 10, 'bold'), padding=6)
        self.style.map('Shift.TButton', background=[('active', '#00838f')])

        self.style.configure('Fold.TButton', foreground='white', background='#6d6e71', font=('Segoe UI', 10, 'bold'), padding=6)
        self.style.map('Fold.TButton', background=[('active', '#424242')])

        self.style.configure('Plot.TButton', foreground='white', background='#1565c0', font=('Segoe UI', 10, 'bold'), padding=6)
        self.style.map('Plot.TButton', background=[('active', '#0d47a1')])

        self.style.configure('Save.TButton', foreground='white', background='#7cb342', font=('Segoe UI', 10, 'bold'), padding=6)
        self.style.map('Save.TButton', background=[('active', '#558b2f')])

        self.style.configure('ClearResult.TButton', foreground='white', background='#ef5350', font=('Segoe UI', 10, 'bold'), padding=6)
        self.style.map('ClearResult.TButton', background=[('active', '#e53935')])

        # build UI
        self.setup_gui()

    def setup_gui(self):
        # plot on left, controls on right
        plot_frame = ttk.Frame(self.root, padding="8")
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ---- Scrollable Control Panel ----
        control_container = ttk.Frame(self.root)
        control_container.pack(side=tk.RIGHT, fill=tk.Y)

        canvas = tk.Canvas(control_container, width=360)
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

        self.signals_listbox = tk.Listbox(mgmt_frame, height=8, selectmode=tk.MULTIPLE, bg="#fafafa", fg="#111", font=('Consolas', 10))
        self.signals_listbox.pack(fill=tk.X, pady=6)

        ops_frame = ttk.LabelFrame(control_frame, text="Operations", padding="6")
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
        ttk.Button(ops_frame, text="Plot Selected", style='Plot.TButton',
                   command=self.plot_selected).pack(fill=tk.X, pady=6)
        ttk.Button(ops_frame, text="Plot Two Signals", style='Plot.TButton',
           command=self.plot_two_selected).pack(fill=tk.X, pady=6)
        
        # Display options (discrete or continuous)
        disp_frame = ttk.LabelFrame(control_frame, text="Display Options", padding=6)
        disp_frame.pack(fill=tk.X, pady=(6, 8), padx=6)

        self.display_mode = tk.StringVar(value='discrete')
        self.fs_for_time_plot = tk.DoubleVar(value=100.0)

        ttk.Radiobutton(disp_frame, text="Discrete", variable=self.display_mode, value='discrete').pack(anchor=tk.W)
        ttk.Radiobutton(disp_frame, text="Continuous", variable=self.display_mode, value='continuous').pack(anchor=tk.W)

        fs_frame = ttk.Frame(disp_frame)
        fs_frame.pack(fill=tk.X, pady=6)
        ttk.Label(fs_frame, text="Time-plot fs (Hz):").pack(side=tk.LEFT)
        self.fs_entry = ttk.Entry(fs_frame, width=8, textvariable=self.fs_for_time_plot)
        self.fs_entry.pack(side=tk.LEFT, padx=6)

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

        # nicer default appearance for plots
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('Index (n)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Signal Visualization')

    def read_signal_file(self, filename):
        """Read signal from file in the specified format"""
        try:
            with open(filename, 'r') as f:
                f.readline()
                f.readline()

                num_samples_line = f.readline().strip()
                if not num_samples_line:
                    raise ValueError("Invalid file format")

                num_samples = int(num_samples_line)
                indices = []
                samples = []

                for _ in range(num_samples):
                    line = f.readline().strip()
                    if not line:
                        break

                    parts = line.split()
                    if len(parts) == 2:
                        indices.append(int(parts[0]))
                        samples.append(float(parts[1]))

                return indices, samples

        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")

    def load_signal(self):
        """Load a signal from file"""
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
        """Get currently selected signals from listbox"""
        selected_indices = self.signals_listbox.curselection()
        return [self.signals[i] for i in selected_indices]

    def add_signals(self):
        """Add selected signals"""
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
        """Subtract selected signals (first minus the rest)"""
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
        """Multiply selected signal by constant"""
        selected_signals = self.get_selected_signals()

        if len(selected_signals) != 1:
            messagebox.showwarning("Warning", "Please select exactly one signal")
            return

        try:
            constant = float(self.const_entry.get())
            signal = selected_signals[0]

            result_samples = [s * constant for s in signal.samples]
            self.result_signal = Signal(signal.indices.copy(), result_samples,
                                        f"Multiplied by {constant}")
            self.plot_result()
            messagebox.showinfo("Success", f"Signal multiplied by {constant} successfully!")

        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for constant")
        except Exception as e:
            messagebox.showerror("Error", f"Multiplication failed: {str(e)}")

    def shift_signal(self):
        """Shift signal by k steps (x(n+k) or x(n-k))"""
        selected_signals = self.get_selected_signals()

        if len(selected_signals) != 1:
            messagebox.showwarning("Warning", "Please select exactly one signal")
            return

        try:
            k = int(self.shift_entry.get())
            signal = selected_signals[0]

            new_indices = [idx - k for idx in signal.indices]

            self.result_signal = Signal(new_indices, signal.samples.copy(),
                                        f"Shifted by {k}")
            self.plot_result()
            messagebox.showinfo("Success", f"Signal shifted by {k} successfully!")

        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer for shift")
        except Exception as e:
            messagebox.showerror("Error", f"Shifting failed: {str(e)}")

    def fold_signal(self):
        """Fold signal (x(-n))"""
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
        """Plot the result signal"""
        if self.result_signal is None:
            return

        self.ax.clear()

        indices, samples = self.result_signal.get_plot_data()
        self.ax.stem(indices, samples, linefmt='r-', markerfmt='ro',
                     basefmt=' ', label=self.result_signal.name)

        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('Index (n)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Operation Result')
        self.ax.legend()

        self.canvas.draw()

    def save_result(self):
        """Save result signal to file"""
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
                # Write header
                f.write("0\n")
                f.write("0\n")
                f.write(f"{len(self.result_signal.indices)}\n")

                # Write samples
                for idx, sample in zip(self.result_signal.indices, self.result_signal.samples):
                    f.write(f"{idx} {sample}\n")

            messagebox.showinfo("Success", f"Result saved to {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")

    def clear_result(self):
        """Clear the result signal"""
        self.result_signal = None
        self.ax.clear()
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('Index (n)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Signal Visualization')
        self.canvas.draw()

    def clear_all(self):
        """Clear all signals and results"""
        self.signals.clear()
        self.result_signal = None
        self.signals_listbox.delete(0, tk.END)
        self.clear_result()
        messagebox.showinfo("Info", "All signals cleared")

    def open_generate_dialog(self, wave_type: str = 'sine'):
        """Open a dialog to generate sine/cosine signals.

        Parameters asked:
          - amplitude A
          - phase theta (degrees)
          - analog frequency f (Hz)
          - sampling frequency fs (Hz)
          - duration T (seconds)

        Enforce Nyquist: fs > 2 * f
        """
        dlg = tk.Toplevel(self.root)
        dlg.title(f'Generate {wave_type.title()} Wave')
        dlg.grab_set()

        entries = {}

        def add_row(parent, label_text, default):
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=4, padx=6)
            ttk.Label(row, text=label_text).pack(side=tk.LEFT)
            ent = ttk.Entry(row)
            ent.pack(side=tk.LEFT, padx=6)
            ent.insert(0, str(default))
            return ent

        entries['A'] = add_row(dlg, 'Amplitude (A):', 1.0)
        entries['theta'] = add_row(dlg, 'Phase (degrees):', 0.0)
        entries['f'] = add_row(dlg, 'Analog frequency f (Hz):', 5.0)
        entries['fs'] = add_row(dlg, 'Sampling frequency fs (Hz):', 50.0)
        entries['T'] = add_row(dlg, 'Duration T (seconds):', 1.0)

        def generate_and_close():
            try:
                A = float(entries['A'].get())
                theta_deg = float(entries['theta'].get())
                f = float(entries['f'].get())
                fs = float(entries['fs'].get())
                T = float(entries['T'].get())

                if fs <= 2 * f:
                    # Nyquist violation
                    if not messagebox.askyesno('Nyquist Warning', f'Chosen sampling frequency fs = {fs} Hz does not satisfy Nyquist (fs > 2*f = {2*f} Hz).\n\nDo you want to continue anyway?'):
                        return

                # generate samples as discrete-time signal: n = 0..N-1, x[n] = A * sin(2*pi*f*(n/fs)+theta)
                N = max(1, int(np.ceil(T * fs)))
                n = np.arange(N)
                theta = np.deg2rad(theta_deg)
                if wave_type == 'sine':
                    x = A * np.sin(2 * np.pi * f * (n / fs) + theta)
                else:
                    x = A * np.cos(2 * np.pi * f * (n / fs) + theta)

                indices = n.tolist()
                samples = x.tolist()
                name = f"{wave_type}_{A}A_{f}Hz_fs{fs}Hz_T{T}s"
                sig = Signal(indices, samples, name)
                self.signals.append(sig)
                self.signals_listbox.insert(tk.END, f"{name} ({len(indices)} samples)")

                dlg.destroy()
                messagebox.showinfo('Success', f'{wave_type.title()} signal generated: {N} samples')

            except Exception as e:
                messagebox.showerror('Error', f'Invalid input: {e}')

        btn = ttk.Button(dlg, text='Generate', command=generate_and_close)
        btn.pack(pady=8)


def main():
    root = tk.Tk()
    app = DSPApplication(root)
    root.mainloop()


if __name__ == "__main__":
    main()
