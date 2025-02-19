import traceback
from PyQt5.QtCore import QRunnable, pyqtSlot, pyqtSignal, QObject
import time

class InferenceWorkerSignals(QObject):
    finished = pyqtSignal(dict)  # emit predicted_points on success
    error = pyqtSignal(str)

class InferenceWorker(QRunnable):
    def __init__(self, df, model, scaler, y_scaler):
        super().__init__()
        self.df = df
        self.model = model
        self.scaler = scaler
        self.y_scaler = y_scaler
        self.signals = InferenceWorkerSignals()
    
    @pyqtSlot()
    def run(self):
        try:
            # The heavy network or computation call happens here.
            # (Replace compute_attributes and predict_points with your actual functions.)
            from avulsionprecursors.archive.v0.based_river_inference import compute_attributes, predict_points
            attributes = compute_attributes(self.df)
            df_with_attributes = self.df.copy()
            # Optionally merge computed attributes back in.
            predicted_points = predict_points(self.df, self.model, self.scaler, self.y_scaler)
            self.signals.finished.emit(predicted_points)
        except Exception as e:
            err_str = f"Inference error: {e}\n{traceback.format_exc()}"
            self.signals.error.emit(err_str) 