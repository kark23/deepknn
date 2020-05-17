# ae-knn

```
from main import *
from tensorflow.examples.tutorials.mnist import input_data

mnist= input_data.read_sata_sets("MNIST_data/", one_hot=False)
x, y= mnist.train.next_batch(mnist.test.num_examples)
tst_x, tst_y= mnist.test.next_batch(mnist.test.num_examples)
aek=aeknn('class', x, y, dtyp='numpy', cnn=True)
aek.train(x, y, save='tmp/mod.ckpt', ratio=0., batch=100, epoch=100)
aek.train(x, y, save='tmp/mod.ckpt', ratio=.99, batch=4000, epoch=100, load='tmp/mod.ckpt')
preds=aek.predict(x, y, tst_x, 5, 'tmp/mod.ckpt')
np.count
```

