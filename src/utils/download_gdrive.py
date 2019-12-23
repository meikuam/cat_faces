import os


def download_gdrive(id, filename):
    """
    python version of
    https://github.com/catalyst-team/catalyst/blob/master/bin/scripts/download-gdrive
    script

    https://stackoverflow.com/questions/48133080/how-to-download-a-google-drive-url-via-curl-or-wget/48133859

    this version uses curl
    """
    cookies_path = "cookies.txt"

    cookie_query = f"""curl -c {cookies_path} -s -L "https://drive.google.com/uc?export=download&id={id}" > /dev/null"""
    os.popen(cookie_query).read()
    download_query = f"""curl -Lb {cookies_path} "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {{print $NF}}' {cookies_path}`&id={id}" -o {filename}"""
    print(download_query)
    os.popen(download_query).read()
    clean_query = f"""rm -rf {cookies_path}"""
    os.popen(clean_query).read()


if __name__ == "__main__":
    id = "10xFg7qXLtJ3Oc6rQyOlumkoaOU1U4PiU"
    filename = "kek.pt"

    download_gdrive(id, filename)
