import os
import requests
import zipfile
from pathlib import Path
import shutil


def download_and_extract_domain(url, domain_name, base_path="../data/DomainNet"):
    """Download and extract a single domain dataset"""
    download_path = f"{domain_name}.zip"
    extract_path = os.path.join(base_path, domain_name)

    try:
        # Create directory
        os.makedirs(extract_path, exist_ok=True)

        print(f"Starting download for {domain_name}...")

        # Download file
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get file size for progress display
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0

        with open(download_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)

                    # Show progress
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\rDownload progress for {domain_name}: {progress:.1f}%", end='', flush=True)

        print(f"\nDownload completed: {download_path}")

        # Extract zip file
        print(f"Extracting {domain_name}...")

        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        # Check if there's a nested folder with the same name and flatten it
        nested_folder = os.path.join(extract_path, domain_name)
        if os.path.exists(nested_folder) and os.path.isdir(nested_folder):
            print(f"Found nested {domain_name} folder, flattening structure...")

            # Move all contents from nested folder to parent
            for item in os.listdir(nested_folder):
                src = os.path.join(nested_folder, item)
                dst = os.path.join(extract_path, item)

                if os.path.exists(dst):
                    if os.path.isdir(dst) and os.path.isdir(src):
                        # Merge directories
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        # Replace file
                        if os.path.isfile(dst):
                            os.remove(dst)
                        shutil.move(src, dst)
                else:
                    shutil.move(src, dst)

            # Remove the now-empty nested folder
            os.rmdir(nested_folder)
            print(f"Flattened nested folder structure for {domain_name}")

        print(f"Extraction completed: {extract_path}")

        # Remove temporary zip file
        os.remove(download_path)
        print(f"Temporary zip file removed for {domain_name}")

        # Check extracted contents
        print(f"Extracted contents for {domain_name}:")
        for item in os.listdir(extract_path):
            item_path = os.path.join(extract_path, item)
            if os.path.isdir(item_path):
                file_count = len([f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))])
                print(f"  ?? {item}/ ({file_count} files)")
            else:
                print(f"  ?? {item}")

        return True

    except requests.exceptions.RequestException as e:
        print(f"Download error for {domain_name}: {e}")
        return False
    except zipfile.BadZipFile:
        print(f"Corrupted zip file for {domain_name}")
        return False
    except Exception as e:
        print(f"Error occurred for {domain_name}: {e}")
        return False
    finally:
        # Clean up temporary files
        if os.path.exists(download_path):
            os.remove(download_path)


def download_label_files(domain_name, base_path="../data/DomainNet"):
    """Download train and test label files for a domain"""
    label_base_urls = [
        "https://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt",  # primary
        "https://csr.bu.edu/ftp/visda/2019/multi-source/txt"              # fallback
    ]
    domain_path = os.path.join(base_path, domain_name)
    os.makedirs(domain_path, exist_ok=True)

    success_count = 0

    for split in ["train", "test"]:
        label_filename = f"{domain_name}_{split}.txt"
        label_path = os.path.join(domain_path, label_filename)
        download_success = False

        for base_url in label_base_urls:
            label_url = f"{base_url}/{label_filename}"
            try:
                print(f"Trying to download {label_filename} from {label_url}...")

                response = requests.get(label_url)
                response.raise_for_status()

                with open(label_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)

                print(f"? {label_filename} downloaded successfully from {label_url}")
                success_count += 1
                download_success = True
                break  # ¼º°øÇßÀ¸¸é ´ÙÀ½ splitÀ¸·Î ³Ñ¾î°¨

            except requests.exceptions.RequestException:
                print(f"? Failed from {label_url}, trying next fallback if available...")

        if not download_success:
            print(f"? All attempts failed to download {label_filename}")

    return success_count == 2



def download_all_domains():
    """Download all DomainNet domains and their label files"""
    # Define all domain URLs and their corresponding folder names
    domains = {
        "clipart": "https://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip",
        "infograph": "https://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
        "painting": "https://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip",
        "quickdraw": "https://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip",
        "real": "https://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
        "sketch": "https://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip"
    }

    successful_downloads = []
    failed_downloads = []
    successful_labels = []
    failed_labels = []

    print("=" * 60)
    print("Starting DomainNet Multi-Domain Dataset Download")
    print("=" * 60)

    for domain_name, url in domains.items():
        print(f"\n{'=' * 50}")
        print(f"Processing domain: {domain_name.upper()}")
        print(f"{'=' * 50}")

        # Download and extract domain images
        success = download_and_extract_domain(url, domain_name)

        if success:
            successful_downloads.append(domain_name)
            print(f"? {domain_name} images completed successfully!")
        else:
            failed_downloads.append(domain_name)
            print(f"? {domain_name} images failed!")

        # Download label files
        print(f"\nDownloading label files for {domain_name}...")
        label_success = download_label_files(domain_name)

        if label_success:
            successful_labels.append(domain_name)
            print(f"? {domain_name} labels completed successfully!")
        else:
            failed_labels.append(domain_name)
            print(f"? {domain_name} labels failed!")

    # Final summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    print("IMAGES:")
    if successful_downloads:
        print(f"? Successfully downloaded ({len(successful_downloads)}):")
        for domain in successful_downloads:
            print(f"   - {domain}")

    if failed_downloads:
        print(f"? Failed downloads ({len(failed_downloads)}):")
        for domain in failed_downloads:
            print(f"   - {domain}")

    print("\nLABELS:")
    if successful_labels:
        print(f"? Successfully downloaded ({len(successful_labels)}):")
        for domain in successful_labels:
            print(f"   - {domain}")

    if failed_labels:
        print(f"? Failed downloads ({len(failed_labels)}):")
        for domain in failed_labels:
            print(f"   - {domain}")

    if len(successful_downloads) == len(domains) and len(successful_labels) == len(domains):
        print(f"\n?? All {len(domains)} domains (images + labels) downloaded successfully!")
        print("DomainNet dataset is ready to use!")
    else:
        print(f"\n??  Some downloads failed.")
        print("Please check your internet connection and try again for failed items.")

    return successful_downloads, failed_downloads, successful_labels, failed_labels


if __name__ == "__main__":
    successful_images, failed_images, successful_labels, failed_labels = download_all_domains()

    print(f"\nFinal Status:")
    print(f"Total domains: 6")
    print(f"Images - Successful: {len(successful_images)}, Failed: {len(failed_images)}")
    print(f"Labels - Successful: {len(successful_labels)}, Failed: {len(failed_labels)}")