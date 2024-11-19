import pickle
from sklearn.decomposition import PCA
import os

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
connections.connect(host='47.102.103.246', port='19530')


def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, descrition='ids', max_length=10000, is_primary=True,
                    auto_id=True),
        FieldSchema(name='filepath', dtype=DataType.VARCHAR, description='filepath', max_length=1024),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, descrition='embedding vectors', dim=dim),
    ]
    schema = CollectionSchema(fields=fields, description='reverse image search')
    collection = Collection(name=collection_name, schema=schema)

    # create IVF_FLAT index for collection.
    index_params = {
        'metric_type': 'L2',
        'index_type': "IVF_FLAT",
        'params': {"nlist": 10000}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection


collection = create_milvus_collection('images_test', 657)


with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)
X = [item['embedding'] for item in embeddings]
pca = PCA(n_components=657)
X = pca.fit_transform(X)
for item, vec in zip(embeddings, X):
    item['embedding'] = vec
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)
with open('pca.pkl', 'wb') as f:
    pickle.dump(pca, f)

index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 10000}
}
with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)
base_path = r"/mnt/workspace/datasets"
# 获取所有图片路径
files = [os.path.join(base_path, file) for file in os.listdir(base_path)]
for item in embeddings:
    collection.insert([
        [item['filepath']],
        [item['embedding']]
    ])

