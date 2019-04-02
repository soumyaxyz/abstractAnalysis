from model.data_utils import Dataset
from model.models import HANNModel
from model.config import Config
import argparse
import os

parser = argparse.ArgumentParser()

def save_res():
  import subprocess
  
  os.system("zip -r results.zip results")

  # Install the PyDrive wrapper & import libraries.
  # This only needs to be done once per notebook.
  os.system("pip install -U -q PyDrive")
  from pydrive.auth import GoogleAuth
  from pydrive.drive import GoogleDrive
  from google.colab import auth
  from oauth2client.client import GoogleCredentials

  # Authenticate and create the PyDrive client.
  # This only needs to be done once per notebook.
  auth.authenticate_user()
  gauth = GoogleAuth()
  gauth.credentials = GoogleCredentials.get_application_default()
  drive = GoogleDrive(gauth)

    

  file_id = '1n0GwvL_YNE6Kr_wpWW2v0oNcaD6vA4Ae'

  uploaded = drive.CreateFile({'id': file_id})
  uploaded.SetContentFile('results.zip')
  uploaded.Upload()
  print('Re-uploaded file with ID {}'.format(uploaded.get('id')))

  os.system("rm results.zip")


def main():
    # create instance of config
    config = Config(parser)

    # build model
    model = HANNModel(config)
    model.build()
    if config.restore:
        model.restore_session("results/test/model.weights/") # optional, restore weights

    # create datasets
    dev   = Dataset(config.filename_dev, config.processing_word,
                    config.processing_tag, config.max_iter)
    train = Dataset(config.filename_train, config.processing_word,
                    config.processing_tag, config.max_iter)
    test  = Dataset(config.filename_test, config.processing_word,
                    config.processing_tag, config.max_iter)

    # train model
    model.train(train, dev,  zip_to_drive=None)

    # evaluate model
    model.restore_session(config.dir_model)
    metrics = model.evaluate(test)

    with open(os.path.join(config.dir_output, 'test_results.txt'), 'a') as file:
        file.write('{}\n'.format(metrics['classification-report']))
        file.write('{}\n'.format(metrics['confusion-matrix']))
        file.write('{}\n\n'.format(metrics['weighted-f1']))

if __name__ == "__main__":
    main()
