import pandas as pd
import os
import shutil
df = pd.read_csv("/home/gyang/MAE-GAN/metadata.csv")
df = df[df['dataset']=='train']
df.to_csv("/home/gyang/MAE-GAN/train.csv")
# df['site_id']=df['site_id']+'_'+df['site'].astype('str')
for row in list(df['site_id']):
    exp,plate_id,wellpos,site = row.split('_')
    # print(exp,plate_id,wellpos,site )
    os.makedirs(f"/media/NAS06/zhc/rxrx1/train_data/{row}",exist_ok=True)
    for channel in range(1,7):
        

        # Define source and destination paths
        source_path = f"/media/NAS06/zhc/rxrx1/images/{exp}/Plate{plate_id}/{wellpos}_s{site}_w{channel}.png"  # Replace with your source image path
        destination_path = f"/media/NAS06/zhc/rxrx1/train_data/{row}/{wellpos}_s{site}_w{channel}.png" # Replace with your destination path
        try:
            # Copy the file
            shutil.copy(source_path, destination_path)
            # print(f"File copied successfully to {destination_path}")
        except FileNotFoundError:
            print(f"{row}")

