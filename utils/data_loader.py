import pandas as pd
import pathlib as pl

def load_data(DATA_PATH = "../data/"):
    def read_data(subdir: str):
        data = [x for x in (pl.Path(DATA_PATH)/subdir).iterdir() if x.suffix == ".csv"]
        data = [pd.read_csv(d) for d in data]
        data = pd.concat(data)
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        return data[["timestamp", "differential_potential_CH1", "differential_potential_CH2", "transpiration"]]

    data_temp = read_data("exp_Temperature")
    data_wind = read_data("exp_Wind")

    return (data_temp, data_wind)