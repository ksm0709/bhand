filepath = input('Filepath : ')
savename = input('Save name : ')
savepath = './tf_data/embedding/embeding/'+savename+'.tsv'

csv = open(filepath,'r')
tsv = open(savepath,'w')


while True:
    line = csv.readline()

    if not line : break

    

csv.close()