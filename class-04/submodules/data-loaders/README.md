# Data Loading Tools

This repository holds tools for loading datasets to be used in RUC's Practical AI class.

## Installation via a Dev Container (Recommended for Development)

If developing on this project, it is recommended to use a locally hosted Dev Container. Codespaces is not recommended since downloading large datasets can consume space (which may be a problem if you want to stay under the free limit for codespaces).

To use this repository via a Dev Container, be sure you have the Dev Containers extension installed, along with Docker Desktop and WSL 2 (Windows only). Clone the repository, open VS Code in the repository root, and click the button shown in the pop up in the bottom-right corner to open in the Dev Container.

If the popup doesn't show type `Cmd` / `Ctrl` + `Shift` + `P` &rarr; `Reopen in Container` (if the container is already built) or `Cmd` / `Ctrl` + `Shift` + `P` &rarr; `Build and Open in Container` if the container is not yet built.

## Usage in a Dev Container

### Using the Python Data Generators via a Devcontainer

@TODO (Mauro Sanchirico): Populate

### Using the NBIA Data Loaders via a Devcontainer

@TODO (Mauro Sanchirico): Populate

## Installation as a Component of Another Project

If using this repository in another project, it is recommended to include it as a git submodule.

Start by making a submodules folder in your project and adding the data-loaders repository as a submodule.

```bash
cd your-project
mkdir submodules
git submodule add https://github.com/ruc-practical-ai/data-loaders.git submodules/data-loaders
```

Some installers use expect scripts. Add expect to your environment by adding the following lines to your setup script, or typing in the terminal.

```bash
sudo apt-get update
sudo apt-get -y install expect
```

The National Biomedical Imaging Archive (NBIA) data retriever is used to pull medical datasets. A script is included for installing it. Add a call to this script to your setup script, or call it in the terminal directly.

```bash
bash submodules/data-loaders/.devcontainer/nbia_data_retriever_install.sh
```

Add a command to your setup script to ensure that you have the latest submodules.

```bash
git submodule update --init --recursive
```

Add the submodule to your poetry project. You only need to do this once if you are committing your pyproject.toml to version control.

```bash
poetry add ./submodules/data-loaders
```

## Usage as a Component of Another Project

### Using the Python Data Generators in Another Project

@TODO (Mauro Sanchirico): Populate

### Using the NBIA Data Loaders via Another Project

@TODO (Mauro Sanchirico): Populate

## Local Installation

For local installation, the process to obtain required dependencies will be machine / environment dependent. For installation on Ubuntu, the `.devcontainer/configure_environment.sh` script can be used.

### Dependencies for Local Installation

It is recommended to use this repository in a locally hosted Dev Container. If attempting to install locally, the following dependencies are required.

#### NBIA Data Retriever

Some tools in this repository use Python to generate datasets or download basic machine learning datasets. Other tools access data from The Cancer Imaging Archive (TCIA) using the National Biomedical Imaging Archives (NBIA) Data Retriever.

The NBIA Data Retriever command line interface is required to run some scripts in this repository. Installation and usage instructions can be found [here](https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever+Command-Line+Interface+Guide).

#### Expect

Scripts in this repository require the [expect](https://linux.die.net/man/1/expect) linux utility. Some scripts require expect to work. Install expect using the preferred method for your system.

#### Poetry

This project is built on Python 3.12. Poetry is required for installation. To install Poetry, view the instructions [here](https://python-poetry.org/docs/).

### Local Installation Steps

The dependencies required for local installation can be found in `.devcontainer/Dockerfile` and `.devcontainer/configure_environment.sh`.

For local installation, perform set up and install dependencies in the order they appear in the Dockerfile and the configuration script, starting with the Dockerfile.

Final setup commands can be found in `.devcontainer/post_attach.sh`.

Note that setup will be different depending on the local OS.

## About the Datasets

### Fundamental Datasets

The `pydatapull` package contains educational code to generate several basic machine learning datasets.

### Online Datasets - Python

The `pydatapull` package also contains code to access several machine learning datasets hosted online, intended for educational and research purposes.

### Online Datasets - The Cancer Imaging Archive (TCIA)

TCIA is an online service which hosts a large archive of medical images of cancer accessible for public download. More information about TCIA can be found on its main webpage, [https://www.cancerimagingarchive.net/](https://www.cancerimagingarchive.net/):

> "The Cancer Imaging Archive (TCIA) is a service which de-identifies and hosts a large publicly available archive of medical images of cancer.  TCIA is funded by the Cancer Imaging Program (CIP), a part of the United States  National Cancer Institute (NCI), and is managed by the Frederick National Laboratory for Cancer Research (FNLCR). The imaging data are organized as “collections” defined by a common disease (e.g. lung cancer), image modality or type (MRI, CT, digital histopathology, etc) or research focus. DICOM is the primary file format used by TCIA for radiology imaging.   An emphasis is made to provide supporting data related to the images such as patient outcomes, treatment details, genomics and expert analyses."

TCIA encourages researchers to publish results obtained on the TCIA collections. See [https://www.cancerimagingarchive.net/about-the-cancer-imaging-archive-tcia/](https://www.cancerimagingarchive.net/about-the-cancer-imaging-archive-tcia/) for additional information.

#### Browse TCIA Collections

You can browse the datasets hosted on TCIA here: [https://www.cancerimagingarchive.net/browse-collections/](https://www.cancerimagingarchive.net/browse-collections/).

#### The National Biomedical Imaging Archive (NBIA) Data Retriever

The NBIA Data Retriever is used to download data from The Cancer Imaging Archive (TCIA). More information on the NBIA Data Retriever can be found [here](https://wiki.nci.nih.gov/display/NBIA).

The NBIA Data Retriever requires a manifest file which specifies which data it should download from TCIA before performing a download.

Manifest files ensure data downloads are repeatable. This repository contains several manifest files in the `nbia_manifest/` directory.

The NBIA Data Retriever has both a command line and a graphical user interface. The tools in `pydatapull` abstract this away however, and users of these tools should not need to interact with the NBIA interface.

For those who do need to interact directly with the NBIA Data Retriever, more information can be found on its CLI [here](https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever+Command-Line+Interface+Guide).

New releases of the tool can be found in the [NBIA GitHub](https://github.com/CBIIT/NBIA-TCIA/releases).

#### TCIA User Guide

Complete instructions on interacting with TCIA can be found in the [user guide](https://wiki.cancerimagingarchive.net/display/NBIA/TCIA+Radiology+Portal+User%27s+Guide).

#### Producing a New Manifest File

The manifest files are produced by browsing the online [TCIA search tool](https://nbia.cancerimagingarchive.net/nbia-search/).

A manifest file can either be downloaded from the TCIA search tool by clicking a download button for a dataset, or the following steps can be used to produce a custom manifest file.

1. Navigate to the TCIA search tool in a web browser: [https://nbia.cancerimagingarchive.net/nbia-search/](https://nbia.cancerimagingarchive.net/nbia-search/).
2. Select the datasets of interest on the sidebar.
3. Add the data products of interest to the shopping cart.
4. Navigate to the `Cart` tab and click `Download`.
5. This will produce a manifest file in your `Downloads` folder (or your browser's default location).

Once you have a manifest file, move it into your development environment (e.g., make a new folder in the `nbia_manifest_files` directory and put the file there).

Once you have placed a manifest file in this environment, the tools in this repository can be used it to download images from TCIA.

## License

This repository is provided with an MIT license. See the `LICENSE` file.

Datasets downloaded with the NBIA Data Retriever will include a license file with the dataset in the `downloads/nbia` folder. These licenses may come with extra constraints per dataset that must be adhered to.

## Contributing

Please email Mauro Sanchirico at ms3978@camden.rutgers.edu (academic) or sanchirico.mauro@gmail.com (personal) with questions, comments, bug reports, or suggestions for improvement.
