import tensorflow as tf
from einops.layers.tensorflow import Rearrange
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv3D, ZeroPadding3D, Softmax, Lambda, Add, Layer
from tensorflow.keras import initializers
from tensorflow import einsum, nn, meshgrid

# helpers functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_rel_pos(n):
    pos = tf.stack(meshgrid(tf.range(n), tf.range(n), indexing = 'ij'))
    pos = Rearrange('n i j -> (i j) n')(pos)             # [n*n, 2] pos[n] = (i, j)
    rel_pos = pos[None, :] - pos[:, None]                # [n*n, n*n, 2] rel_pos[n, m] = (rel_i, rel_j)
    rel_pos += n - 1                                     # shift value range from [-n+1, n-1] to [0, 2n-2]
    return rel_pos

# lambda layer

class LambdaLayer(Layer):
    def __init__(
        self,
        *,
        dim_k,
        n = None,
        r = None,
        heads = 4,
        dim_out = None,
        dim_u = 1):
        super(LambdaLayer, self).__init__()

        self.out_dim = dim_out
        self.u = dim_u  # intra-depth dimension
        self.heads = heads

        assert (dim_out % heads) == 0, 'values dimension must be divisible by number of heads for multi-head query'
        self.dim_v = dim_out // heads
        self.dim_k = dim_k
        self.heads = heads

        self.to_q = Conv2D(self.dim_k * heads, 1, use_bias=False)
        self.to_k = Conv2D(self.dim_k * dim_u, 1, use_bias=False)
        self.to_v = Conv2D(self.dim_v * dim_u, 1, use_bias=False)

        self.norm_q = BatchNormalization()
        self.norm_v = BatchNormalization()

        self.local_contexts = exists(r)
        if exists(r):
            assert (r % 2) == 1, 'Receptive kernel size should be odd'
            self.pos_conv = Conv3D(dim_k, (1, r, r), padding='same')
        else:
            assert exists(n), 'You must specify the window length (n = h = w)'
            rel_length = 2 * n - 1
            self.rel_pos_emb = self.add_weight(name='pos_emb',
                                               shape=(rel_length, rel_length, dim_k, dim_u),
                                               initializer=initializers.random_normal,
                                               trainable=True)
            self.rel_pos = calc_rel_pos(n)

    def get_config(self):
        base_config = super().get_config()
        return base_config

    def call(self, x, **kwargs):
        """
        x : (1, 8, 8, 1280)
        k=320
        r = 3
        heads=4
        out_dims=1280
        """
        b, hh, ww, c, u, h = *x.get_shape().as_list(), self.u, self.heads
        
        q = self.to_q(x) # (1, 8, 8, 1280) 
        k = self.to_k(x) # (1, 8, 8, 320)
        v = self.to_v(x) # (1, 8, 8, 320) 

        q = self.norm_q(q)
        v = self.norm_v(v)
        #print(q.shape, k.shape, v.shape, b, hh, ww, c, self.dim_v)
        q = tf.reshape(q, (-1, h, self.dim_v, hh*ww))
        k = tf.reshape(k, (-1, u, self.dim_v, hh*ww))
        v = tf.reshape(v, (-1, u, self.dim_v, hh*ww))
        #print(q.shape)
        #q = Rearrange('b hh ww (h k) -> b h k (hh ww)', h=h)(q) # (1, 4, 320, 64) 
        #k = Rearrange('b hh ww (u k) -> b u k (hh ww)', u=u)(k) # (1, 1, 320, 64)
        #v = Rearrange('b hh ww (u v) -> b u v (hh ww)', u=u)(v) # (1, 1, 320, 64)

        k = nn.softmax(k)
        #print(q.shape, k.shape, v.shape)
        
        Lc = einsum('b u k m, b u v m -> b k v', k, v) # (1, 320, 320) 
        Yc = einsum('b h k n, b k v -> b n h v', q, Lc) # (1, 64, 4, 320)
        #print(Lc.shape, Yc.shape)
        if self.local_contexts:
            v = Rearrange('b u v (hh ww) -> b v hh ww u', hh=hh, ww=ww)(v) # (1, 320, 8, 8, 1)
            Lp = self.pos_conv(v) # (1, 320, 8, 8, 320) 
            #print(v.shape, Lp.shape)
            Lp = Rearrange('b v h w k -> b v k (h w)')(Lp) # (1, 320, 320, 64)
            #print(Lp.shape)
            Yp = einsum('b h k n, b v k n -> b n h v', q, Lp) # (1, 64, 4, 320)
            #print(Yp.shape)
        else:
            rel_pos_emb = tf.gather_nd(self.rel_pos_emb, self.rel_pos)
            Lp = einsum('n m k u, b u v m -> b n k v', rel_pos_emb, v)
            Yp = einsum('b h k n, b n k v -> b n h v', q, Lp)
        Y = Yc + Yp # (1, 64, 4, 320) + (1, 64, 4, 320) = > (1, 64, 4, 320) 
        out = Rearrange('b (hh ww) h v -> b hh ww (h v)', hh = hh, ww = ww)(Y)
        #print(Y.shape, Yc.shape, Yp.shape, out.shape)
        return out # (1, 8, 8, 1280)
    
    def compute_output_shape(self, input_shape):
        return (*input_shape[:2], self.out_dim)


