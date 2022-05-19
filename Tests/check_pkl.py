from Dock2D.DatasetGeneration.ProteinPool import ProteinPool

if __name__ == "__main__":

    trainpoolname = '../DatasetGeneration/PoolData/trainvalidset_protein_pool400.pkl'
    testpoolname = '../DatasetGeneration/PoolData/testset_protein_pool400.pkl'

    trainpool = ProteinPool.load(trainpoolname)
    testpool = ProteinPool.load(testpoolname)

    print(trainpool.proteins)
    print(trainpool.params)
    print(testpool.proteins)
    print(testpool.params)

    # trainpool.save(trainpoolname)
    # testpool.save(testpoolname)
