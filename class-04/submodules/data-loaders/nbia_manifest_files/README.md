# NBIA Manifest Files

This folder contains preconfigured National Biomedical Imaging Archive (NBIA) Data Retriever manifest files. More information on the NBIA Data Retriever can be found [here](https://wiki.nci.nih.gov/display/NBIA).

The NBIA Data Retriever is used to download data from The Cancer Imaging Archive (TCIA). More information on TCIA can be found [here](https://www.cancerimagingarchive.net/).

## Using an Existing Manifest File

To use an existing manifest file, follow the instructions for the tool being used to download it.

### Downloading with the NBIA Data Retriever Directly

To download the data referenced by an existing manifest file with the NBIA Data Retriever directly follow the instructions [here](https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever+Command-Line+Interface+Guide).

### Downloading with a Script

A set of convenience scripts are provided to download data referenced in manifest files. The scripts must be run with `expect` and assume `expect` is already installed.

#### For the First Time

If downloading a new dataset for the first time, use the `scripts/nbia_download.sh` script.

For example, from the root of this repository, run:

```bash
expect ./scripts/nbia_download.sh /workspaces/data-loaders/nbia_manifest_files/small_test_manifest/manifest-1727052249988.tcia /workspaces/data-loaders/downloads/nbia
```

General usage:

```bash
expect ./scripts/nbia_download.sh PATH/TO/MANIFEST.tcia PATH/TO/DOWNLOAD_LOCATION
```

#### Resuming a Download

If resuming a scripted download, use the `scripts/nbia_resume_download.sh` script.

For example, from the root of this repository, run:

```bash
expect ./scripts/nbia_resume_download.sh /workspaces/data-loaders/nbia_manifest_files/small_test_manifest/manifest-1727052249988.tcia /workspaces/data-loaders/downloads/nbia
```

General usage:

```bash
expect ./scripts/nbia_resume_download.sh PATH/TO/MANIFEST.tcia PATH/TO/DOWNLOAD_LOCATION
```

#### Re-downloading (e.g., Re-downloading a Corrupted Dataset)

If re-downloading a dataset (e.g., one that was corrupted) use the `scripts/nbia_redownload.sh` script.

```bash
expect ./scripts/nbia_redownload.sh /workspaces/data-loaders/nbia_manifest_files/small_test_manifest/manifest-1727052249988.tcia /workspaces/data-loaders/downloads/nbia
```

General usage:

```bash
expect ./scripts/nbia_redownload.sh PATH/TO/MANIFEST.tcia PATH/TO/DOWNLOAD_LOCATION
```

## Producing a New Manifest File

The manifest files are produced by browsing the online [TCIA search tool](https://nbia.cancerimagingarchive.net/nbia-search/).

A manifest file can either be downloaded from the TCIA search tool by clicking a download button for a dataset, or the following steps can be used to produce a custom manifest file.

1. Navigate to the TCIA search tool in a web browser: [https://nbia.cancerimagingarchive.net/nbia-search/](https://nbia.cancerimagingarchive.net/nbia-search/).
2. Select the datasets of interest on the sidebar.
3. Add the data products of interest to the shopping cart.
4. Navigate to the `Cart` tab and click `Download`.
5. This will produce a manifest file in your `Downloads` folder (or your browser's default location).

Once you have a manifest file, move it into your development environment (e.g., make a new folder in the `nbia_manifest_files` directory and put the file there).

Once you have placed a manifest file in this environment, the tools in this repository can be used it to download images from TCIA.

## Available Manifest Files

This repository contains several manifest files ready to use for downloading data. When data is downloaded using the tools in this repository, a `LICENSE` file is automatically placed in the corresponding directory in `downloads/nbia`. For each dataset, refer to the corresponding `LICENSE` file for its terms of use.

### Small Test Manifest: Lung Phantom Dataset (Zhao, B.)

A small test manifest is provided to enable trying out the tools in this repository without downloading large datasets.

The test manifest downloads CT scans from the Lung Phantom dataset by Zhao, B. More information about this dataset can be found [here](https://www.cancerimagingarchive.net/collection/lung-phantom/).

Cite this dataset as follows.

> B. Zhao, “Lung Phantom.” The Cancer Imaging Archive, 2015, doi: 10.7937/K9/TCIA.2015.08A1IXOO.

## Contributing

If adding a new manifest file and committing to this repository, please add some brief documentation on it here, including proper citation to its source, and a rough estimate of how large the data products are once downloaded.

### Citations

**Most datasets in TCIA require proper citation and attribution.** Please cite the source of any manifest files here in this README.

### Note to Use Manifest Files for Public Datasets Only

TCIA will require a user name and a password for nonpublic datasets. This repository is not configured to handle non-public datasets. **Do not commit manifest files for non-public datasets to this repository.**