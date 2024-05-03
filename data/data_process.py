import json
'''
json_file='./thumos14.json'
with open(json_file, 'r') as fid:
    json_data = json.load(fid)
json_db = json_data['database']
new_data=dict()
for key in json_db.keys():
    vid_data=json_db[key]
    annots=vid_data["annotations"]
    for ann in annots:
        if ann['label_id']==4:
            new_data[key]=vid_data
            break
for key in new_data.keys():
    flag=False
    vid_data =new_data[key]
    annots = vid_data["annotations"]
    new_annots=[]
    for ann in annots:
        if ann['label_id'] != 7:
            new_annots.append(ann)
    new_data[key]["annotations"]=new_annots
for key in new_data.keys():
    vid_data = new_data[key]
    json_db[key]=vid_data
json_file_our='./thumos14_our.json'
new_j={"version": "Thumos14-30fps", "database":json_db}
with open(json_file_our, "w") as out:
    json.dump(new_j, out)
'''
json_file='./thumos14_our.json'
with open(json_file, 'r') as fid:
    json_data = json.load(fid)
json_db = json_data['database']
new_data=dict()
for key in json_db.keys():
    flag=False
    vid_data=json_db[key]
    annots=vid_data["annotations"]
    for ann in annots:
        if ann['label_id']==4:
            flag=True
            break
    if flag:
        new_data[key]=json_db[key]
for key in new_data.keys():
    json_db[key+'-ad']=new_data[key]
json_file_our='./thumos14_our_ad.json'
new_j={"version": "Thumos14-30fps", "database":json_db}
with open(json_file_our, "w") as out:
    json.dump(new_j, out)