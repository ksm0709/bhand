import tensorflow as tf
from os.path import join
from tensorflow.contrib.tensorboard.plugins import projector

n_embed = 10000
n_data = 8 + 16 + 10 
LOG_DIR = '/home/taeho/catkin_ws/src/bhand/src/tf_log/embedding'

with tf.Session() as sess:
    
    summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

    embedding_var = tf.Variable(tf.random_normal(n_embed, n_data), name='word_embedding')

    sess.run(embedding_var.initializer)
    
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = join(LOG_DIR, 'normal.tsv')

   
    projector.visualize_embeddings(summary_writer, config) 

    saver_embed = tf.train.Saver([embedding_var])
    saver_embed.save(sess, LOG_DIR+'/embedding.ckpt',1)

summary_writer.close()

    
