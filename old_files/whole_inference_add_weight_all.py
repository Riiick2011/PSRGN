import os

cmd0='python fastai_train_add_weight_all.py ./configs/thumos_i3d.yaml'
os.system(cmd0)


def read_file(file):
    f1=open(file,'r')
    s=f1.read()
    m=s.split('\n')
    sorce_list=[]
    for i in range(len(m)-1):
        p=m[i].split(' ')
        sorce_list.append([float(p[2]),float(p[1])])
    sorce_list=sorted(sorce_list)
    sigma=sorce_list[-1][1]
    score=sorce_list[-1][0]
    return sigma,score
ap=[]
if os.path.exists('./test_results/mid_results.txt'):
    os.system('rm -rf ./test_results/mid_results.txt')
for j in range(30,61):
    sigma=j/100.0
    print(j)
    cmd = 'python com_inference_add_weight_all.py ./configs/thumos_i3d.yaml ./models/epoch_%d_model_GCN.pth'%j#+' --sigma '+str(sigma)
    os.system(cmd)
    _,score=read_file('./test_results/mid_results.txt')
    #print(score)
    cmd2 = 'rm -rf ' + './test_results/mid_results.txt'
    os.system(cmd2)
    ap.append([score,j])
ap=sorted(ap)
print(ap)
max_sorce=ap[-1][0]
new_flie=[]
new_flie.append(ap[-1][1])
count=-2
length=len(ap)*-1
while max_sorce-ap[count][0]<0.005 and len(new_flie)<3:
    new_flie.append(ap[count][1])
    count=count-1
ap1=[]
print(new_flie)
for i in new_flie:
    for s in range(11,35):
        sigma=s/100.0
        print(sigma,i)
        cmd1 = 'python com_inference_add_weight_all.py ./configs/thumos_i3d.yaml ./models/epoch_%d_model_GCN.pth'%i+' --sigma '+str(sigma)
        os.system(cmd1)
    sigma1, score = read_file('./test_results/mid_results.txt')
    ap1.append([score, sigma1,i])
    cmd2 = 'rm -rf ' + './test_results/mid_results.txt'
    os.system(cmd2)
ap1=sorted(ap1)
model_name='epoch_%d_model_GCN.pth'%int(ap1[-1][2])
new_name='epoch_%d_%.4f_%.2f_model_GCN_add_weight_all.pth'%(int(ap1[-1][2]),float(ap1[-1][0]),float(ap1[-1][1]))
cmd3='cp ./models/'+model_name+' ./mid_models/'+new_name
os.system(cmd3)
'''
for i in [50]:
    for s in range(20,41):
        sigma=s/100.0
        print(sigma,i)
        cmd1 = 'python com_inference.py ./configs/thumos_i3d.yaml ./models/epoch_%d_model_GCN.pth'%i+' --sigma '+str(sigma)
        os.system(cmd1)

'''