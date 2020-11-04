import pickle
for i in range(14):
    for j in [0, 1]:
        with open('/home/anant/Projects/flatland-challenge-starter-kit-master/scratch/all tests/Test_'+str(i)+'/Level_'+str(j)+'.pkl', 'rb') as f:
            ff=pickle.load(f)
        print('Test', i, 'Level', j, 'agents', len(ff['agents']), 'env_size', len(ff['grid']), len(ff['grid'][0]))
print('DONE')
