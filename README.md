# River Avulsion Precursors Encoded in Alluvial Ridge Geometry

This repository contains companion tools and analysis code for the GRL paper *"River Avulsion Precursors Encoded in Alluvial Ridge Geometry"*. It provides both a processing pipeline and an interactive GUI for analyzing river cross–section data.

---

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/jameshgrn/avulsionprecursors
   cd avulsionprecursors
   ```

2. **Install Dependencies with Poetry**

   ```bash
   poetry install
   ```

   *Alternatively, use your preferred virtual environment mechanism to install dependencies based on the `pyproject.toml` file.*

3. **Configure Environment Variables**

   Create a `.env` file in the repository root with the following (adjust the values as needed):

   ```ini
   GEE_SERVICE_ACCOUNT=your_service_account
   GEE_CREDENTIALS_PATH=path/to/your/credentials.json
   DB_NAME=your_database_name
   DB_USER=your_database_user
   DB_PASSWORD=your_database_password
   DB_HOST=your_database_host
   DB_PORT=your_database_port
   ```

---

## Database Setup

The project uses the SWORD v16 dataset, available [here](https://drive.google.com/drive/folders/14MLBRuqqB3k0K8iAkDEd7XrhqS3_77jv), which we have downloaded and hosted on a PostgreSQL database.

To set up your own PostgreSQL instance:

1. **Install PostgreSQL:**  
   Follow the installation instructions available at [PostgreSQL Documentation](https://www.postgresql.org/docs/current/tutorial-install.html).

2. **Create a New Database and Import the Dataset:**  
   Create a database for the SWORD v16 dataset and import the data into it. Ensure that the dataset is accessible by your application.

3. **Configure Database Credentials:**  
   Update your `.env` file with the following variables:

   ```ini
   DB_NAME=your_database_name
   DB_USER=your_database_user
   DB_PASSWORD=your_database_password
   DB_HOST=your_database_host
   DB_PORT=your_database_port
   ```

For detailed instructions on setting up your own PostgreSQL instance, refer to [this guide](https://www.codecademy.com/article/installing-and-using-postgresql-locally).

---

## How to Run

### Pipeline Mode

The pipeline processes river cross–section data (up through the DEM/hydraulic stages) and saves the results.

Run the pipeline with:

```bash
poetry run python main.py
```

> **Note:** Running `main.py` executes the full processing chain (initializing GEE, connecting to the database, and executing the labeling pipeline). Processed data are saved in the `data` directory.

### GUI Mode

The interactive GUI is provided to view and edit cross–section data.

Launch the GUI using the CLI:

```bash
poetry run python avulsionprecursors/gui/gui_edit_working.py --river VENEZ_2023
```

Replace `VENEZ_2023` with the desired river name. This will open the Cross Section Viewer with the specified river loaded.

> **Important:** The main processing pipeline and the GUI are decoupled. The pipeline (invoked via `main.py`) does not call the GUI.

---

## Code Overview

- **Pipeline Processing:**  
  Implemented in `avulsionprecursors/pipeline/labeling.py`, this module processes the DEM/hydraulic portion of the data for labeling.

- **Database Connectivity:**  
  Database configuration is handled in `avulsionprecursors/db/config.py` and `avulsionprecursors/db/sword.py`.

- **Graphical User Interface (GUI):**  
  The GUI is defined in `avulsionprecursors/gui/gui_edit_working.py` and offers interactive plotting, editing, and updating of cross–section data. It accepts a `--river` argument when launched directly.

---

## Citation

If you use this code in your research, please cite:

*River Avulsion Precursors Encoded in Alluvial Ridge Geometry*, GRL.

---

## License

This project is licensed under MIT.

---


