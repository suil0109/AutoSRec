import pandas as pd

import os
from abc import ABC, abstractmethod
from urllib.parse import urlparse

import bz2
import gzip
import logging
import os
import shutil
import tarfile
import urllib.request
from zipfile import ZipFile

from tqdm import tqdm

logger = logging.getLogger(__name__)


class DownloadProgressBar(tqdm):
    """ From https://stackoverflow.com/a/53877507/8810037
    """
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    try:
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        logger.info("Download successful")
    except:
        logger.exception("Download failed, please try again")
        if output_path.exists():
            os.remove(output_path)


def extract(filepath, out):
    """ out: a file or a directory
    """
    logger.info("Unzipping...")
    filename = filepath.as_posix()

    if filename.endswith(".zip") or filename.endswith(".ZIP"):
        with ZipFile(filepath) as zipObj:
            zipObj.extractall(out)

    elif filename.endswith(".tar.gz"):
        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(out)

    elif filename.endswith(".tar.bz2"):
        with tarfile.open(filepath, "r:bz2") as tar:
            tar.extractall(out)

    elif filename.endswith(".gz"):
        with gzip.open(filepath, "rb") as fin:
            with open(out, "wb") as fout:
                shutil.copyfileobj(fin, fout)

    elif filename.endswith(".bz2"):
        with bz2.open(filepath, "rb") as fin:
            with open(out, "wb") as fout:
                shutil.copyfileobj(fin, fout)
    else:
        logger.error("Unrecognized compressing format of %s", filepath)
        return

    logger.info("OK")

class Dataset(ABC):
    """ Base dataset of SR datasets
    """
    def __init__(self, rootdir):
        """ `rootdir` is the directory of the raw dataset """
        self.rootdir = rootdir

    @property
    def rawpath(self):
        if hasattr(self, "__url__"):
            return self.rootdir.joinpath(os.path.basename(urlparse(self.__url__).path))
        else:
            return ""

    @abstractmethod
    def download(self):
        """ Download and extract the raw dataset """
        pass

    @abstractmethod
    def transform(self):
        """ Transform to the general data format, which is
        a pd.DataFrame instance that contains three columns: [user_id, item_id, timestamp]
        """
        pass

class Amazon(Dataset):

    __corefile__ = {
        "Books": "ratings_Books.csv",
        "Electronics": "ratings_Electronics.csv",
        "Movies": "ratings_Movies_and_TV.csv",
        "CDs": "ratings_CDs_and_Vinyl.csv",
        "Clothing": "ratings_Clothing_Shoes_and_Jewelry.csv",
        "Home": "ratings_Home_and_Kitchen.csv",
        "Kindle": "ratings_Kindle_Store.csv",
        "Sports": "ratings_Sports_and_Outdoors.csv",
        "Phones": "ratings_Cell_Phones_and_Accessories.csv",
        "Health": "ratings_Health_and_Personal_Care.csv",
        "Toys": "ratings_Toys_and_Games.csv",
        "VideoGames": "ratings_Video_Games.csv",
        "Tools": "ratings_Tools_and_Home_Improvement.csv",
        "Beauty": "ratings_Beauty.csv",
        "Apps": "ratings_Apps_for_Android.csv",
        "Office": "ratings_Office_Products.csv",
        "Pet": "ratings_Pet_Supplies.csv",
        "Automotive": "ratings_Automotive.csv",
        "Grocery": "ratings_Grocery_and_Gourmet_Food.csv",
        "Patio": "ratings_Patio_Lawn_and_Garden.csv",
        "Baby": "ratings_Baby.csv",
        "Music": "ratings_Digital_Music.csv",
        "MusicalInstruments": "ratings_Musical_Instruments.csv",
        "InstantVideo": "ratings_Amazon_Instant_Video.csv"
    }

    url_prefix = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"

    def download(self, category):
        download_url(self.url_prefix + self.__corefile__[category], self.rootdir.joinpath(self.__corefile__[category]))

    def transform(self, category, rating_threshold):
        """ Records with rating less than `rating_threshold` are dropped
        """
        df = pd.read_csv(self.rootdir.joinpath(self.__corefile__[category]),
                         header=None,
                         names=["user_id", "item_id", "rating", "timestamp"])
        df = df[df["rating"] >= rating_threshold].drop("rating", axis=1)
        return df



# import sys
# sys.path.append('../../')


import gzip
import json

def parse(path):
  # g = gzip.open(path, 'rb')
  g = json.load(path)
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

# df = getDF('/Users/suillee/Desktop/Research/AutoRec_/autorec_sync_ts/AutoRec-sync/examples/example_datasets/amazon/Video_Games_5.json')


file_path = '../../examples/example_datasets/amazon/Movies_and_TV_5.json'

# f = open('../../examples/example_datasets/amazon/Movies_and_TV_5.json', )

import glob
file=glob.glob(file_path)

review=[]
with open(file[0]) as data_file:
    data=data_file.read()
    for i in data.split('\n'):
        review.append(i)

reviewDataframe = []
for x in review:
    try:
        jdata = json.loads(x)
        reviewDataframe.append((jdata['reviewerID'], jdata['asin'], jdata['reviewerName'], jdata['helpful'][0],
                                jdata['helpful'][1], jdata['reviewText'], jdata['overall'], jdata['summary'],
                                jdata['unixReviewTime'], jdata['reviewTime']))
    except:
        pass

    # Creating a dataframe using the list of Tuples got in the previous step.
dataset = pd.DataFrame(reviewDataframe,
                       columns=['Reviewer_ID', 'Asin', 'Reviewer_Name', 'helpful_UpVote', 'Total_Votes', 'Review_Text',
                                'Rating', 'Summary', 'Unix_Review_Time', 'Review_Time'])

dataset = dataset[['Reviewer_ID', 'Asin', 'Reviewer_Name', 'helpful_UpVote', 'Total_Votes',
                                'Rating', 'Unix_Review_Time', 'Review_Time']]
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

reframed = dataset.sort_values(by=['Reviewer_ID'], ascending=(True))
counts = reframed['Reviewer_ID'].value_counts(dropna=False)

print(counts)