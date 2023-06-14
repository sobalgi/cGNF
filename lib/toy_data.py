# <Source: https://github.com/rtqichen/ffjord/blob/master/lib/toy_data.py >

import numpy as np
import sklearn
import torch
import sklearn.datasets
from PIL import Image
import os

# Dataset iterator
def inf_train_gen(data, rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()
        #print(rng)

    if data == "2spirals-8gaussians":
        data1 = inf_train_gen("2spirals", rng=rng, batch_size=batch_size)
        data2 = inf_train_gen("8gaussians", rng=rng, batch_size=batch_size)
        return np.concatenate([data1, data2], axis=1)

    if data == "4-2spirals-8gaussians":
        data1 = inf_train_gen("2spirals", rng=rng, batch_size=batch_size)
        data2 = inf_train_gen("8gaussians", rng=rng, batch_size=batch_size)
        data3 = inf_train_gen("2spirals", rng=rng, batch_size=batch_size)
        data4 = inf_train_gen("8gaussians", rng=rng, batch_size=batch_size)
        return np.concatenate([data1, data2, data3, data4], axis=1)

    if data == "8-2spirals-8gaussians":
        data1 = inf_train_gen("4-2spirals-8gaussians", rng=rng, batch_size=batch_size)
        data2 = inf_train_gen("4-2spirals-8gaussians", rng=rng, batch_size=batch_size)
        return np.concatenate([data1, data2], axis=1)

    if data == "8-MIX":
        data1 = inf_train_gen("2spirals", rng=rng, batch_size=batch_size)
        data2 = inf_train_gen("8gaussians", rng=rng, batch_size=batch_size)
        data3 = inf_train_gen("swissroll", rng=rng, batch_size=batch_size)
        data4 = inf_train_gen("circles", rng=rng, batch_size=batch_size)
        data8 = inf_train_gen("moons", rng=rng, batch_size=batch_size)
        data6 = inf_train_gen("pinwheel", rng=rng, batch_size=batch_size)
        data7 = inf_train_gen("checkerboard", rng=rng, batch_size=batch_size)
        data5 = inf_train_gen("line", rng=rng, batch_size=batch_size)
        std = np.array([1.604934 , 1.584863 , 2.0310535, 2.0305095, 1.337718 , 1.4043778,  1.6944685, 1.6935346,
                        1.7434783, 1.0092416, 1.4860426, 1.485661 , 2.3067558, 2.311637 , 1.4430547, 1.4430547], dtype=np.float32)
        data = np.concatenate([data1, data2, data3, data4, data5, data6, data7, data8], axis=1)

        return data/std
    if data == "7-MIX":
        data1 = inf_train_gen("2spirals", rng=rng, batch_size=batch_size)
        data2 = inf_train_gen("8gaussians", rng=rng, batch_size=batch_size)
        data3 = inf_train_gen("swissroll", rng=rng, batch_size=batch_size)
        data4 = inf_train_gen("circles", rng=rng, batch_size=batch_size)
        data5 = inf_train_gen("moons", rng=rng, batch_size=batch_size)
        data6 = inf_train_gen("pinwheel", rng=rng, batch_size=batch_size)
        data7 = inf_train_gen("checkerboard", rng=rng, batch_size=batch_size)
        std = np.array([1.604934 , 1.584863 , 2.0310535, 2.0305095, 1.337718 , 1.4043778,  1.6944685, 1.6935346,
                        1.7434783, 1.0092416, 1.4860426, 1.485661 , 2.3067558, 2.311637], dtype=np.float32)
        data = np.concatenate([data1, data2, data3, data4, data5, data6, data7], axis=1)

        return data/std


    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data

    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data == "moons":
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        data = data.astype("float32")
        return data

    elif data == "8gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset

    elif data == "2gaussians":
        scale = 4.
        centers = [(.5, -.5), (-.5, .5)]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * .75
            idx = rng.randint(2)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        #dataset /= 1.414
        return dataset

    elif data == "4gaussians":
        scale = 4.
        centers = [(.5, -.5), (-.5, .5), (.5, .5), (-.5, -.5)]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * .75
            idx = rng.randint(4)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        # dataset /= 1.414
        return dataset

    elif data == "2igaussians":
        scale = 4.
        centers = [(.5, 0.), (-.5, .0)]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * .75
            idx = rng.randint(2)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        # dataset /= 1.414
        return dataset

    elif data == "conditionnal8gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        context = np.zeros((batch_size, 8))
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            context[i, idx] = 1
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset, context

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations)).astype("float32")

    elif data == "2spirals":
        n = np.sqrt(rng.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + rng.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + rng.rand(batch_size // 2, 1) * 0.5 
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += rng.randn(*x.shape) * 0.1
        return x.astype("float32")

    elif data == "2spirals_u":
        # x1 <- z1 -> U <- z2 -> x2 
        z2 = torch.distributions.Normal(0., 4.).sample((batch_size // 2, 1))
        z2 *= z2.numpy()*2
        z1 = torch.distributions.Normal(0., 1.).sample((batch_size // 2, 1))
        z1 *= z1.numpy()*3

        n = np.sqrt(2*z1+3*z2) * 540 * (2 * np.pi) / 360
#         n = np.sqrt(rng.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + rng.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + rng.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += rng.randn(*x.shape) * 0.1
        return x.astype("float32")

    elif data == "2spirals_u_12":
        # x2 <- x1 <- z1 -> U <- z2 -> x2 
        z2 = torch.distributions.Normal(0., 4.).sample((batch_size // 2, 1))
        z2 *= z2.numpy()*2
        z1 = torch.distributions.Normal(0., 1.).sample((batch_size // 2, 1))
        z1 *= z1.numpy()*3

        n = np.sqrt(2*z1+3*z2) * 540 * (2 * np.pi) / 360
#         n = np.sqrt(rng.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + rng.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + rng.rand(batch_size // 2, 1) * 0.5 + d1x**2
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += rng.randn(*x.shape) * 0.1
        return x.astype("float32")

    elif data == "2spirals_u_21":
        # x1 <- z1 -> U <- z2 -> x2 -> x1
        z2 = torch.distributions.Normal(0., 4.).sample((batch_size // 2, 1))
        z2 *= z2.numpy()*2
        z1 = torch.distributions.Normal(0., 1.).sample((batch_size // 2, 1))
        z1 *= z1.numpy()*3

        n = np.sqrt(2*z1+3*z2) * 540 * (2 * np.pi) / 360
#         n = np.sqrt(rng.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1y = np.sin(n) * n + rng.rand(batch_size // 2, 1) * 0.5
        d1x = -np.cos(n) * n + rng.rand(batch_size // 2, 1) * 0.5 + d1y**2
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += rng.randn(*x.shape) * 0.1
        return x.astype("float32")

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1).astype("float32") * 2

    elif data == "line":
        x = rng.rand(batch_size)
        #x = np.arange(0., 1., 1/batch_size)
        x = x * 5 - 2.5
        y = x #- x + rng.rand(batch_size)
        return np.stack((x, y), 1).astype("float32")
    elif data == "line-noisy":
        x = rng.rand(batch_size)
        x = x * 5 - 2.5
        y = x + rng.randn(batch_size)
        return np.stack((x, y), 1).astype("float32")
    elif data == "cos":
        x = rng.rand(batch_size) * 6 - 3
        y = np.sin(x*5) * 2.5 + np.random.randn(batch_size) * .3
        return np.stack((x, y), 1).astype("float32")
    elif data == "joint_gaussian":
        x2 = torch.distributions.Normal(0., 4.).sample((batch_size, 1))
        x1 = torch.distributions.Normal(0., 1.).sample((batch_size, 1)) + (x2**2)/4

        return torch.cat((x1, x2), 1)
#         return torch.cat((x2, x1), 1)
    elif data == "joint_gaussian_12":
        confounder = torch.distributions.Normal(0., 4.).sample((batch_size, 1))
        x2 = torch.distributions.Normal(0., 4.).sample((batch_size, 1))
        x1 = torch.distributions.Normal(0., 1.).sample((batch_size, 1)) + (x2**2)/4

        return torch.cat((x1, x2), 1)
#         return torch.cat((x2, x1), 1)


    elif data == "cont_disc":
        from scipy.stats import norm
        import torch
        from torch.distributions.normal import Normal
        from torch.distributions.bernoulli import Bernoulli
        mu_C1 = -1.0
        sig_C1 = 0.5

        C1 = Normal(torch.tensor([mu_C1]), torch.tensor([sig_C1]))
        c1 = C1.sample(sample_shape=torch.Size([batch_size]))  # normally distributed with loc=0 and scale=1

#         p_A1 = norm.cdf(0.4*c1)
#         p_A1 = torch.from_numpy(p_A1)

#         A1 = Bernoulli(p_A1)
#         a1 = A1.sample(sample_shape=torch.Size([1]))  # normally distributed with loc=0 and scale=1
#         a1.squeeze_(0)#1

        
#         pmf = torch.tensor([0.1, 0.9])#2
#         pmf = torch.tensor([0.25, 0.75])#3
#         pmf = torch.tensor([0.3, 0.7])
#         pmf = torch.tensor([0.4, 0.6])
#         pmf = torch.tensor([0.5, 0.5])#4
#         pmf = torch.tensor([0.6, 0.4])
#         pmf = torch.tensor([0.7, 0.3])
#         pmf = torch.tensor([0.75, 0.25])#5
#         pmf = torch.tensor([0.9, 0.1])#6

#         pmf = torch.tensor([0.2, 0.3, 0.5])#7
#         pmf = torch.tensor([0.3, 0.5, 0.2])#8
#         pmf = torch.tensor([0.5, 0.2, 0.3])#9

#         pmf = torch.tensor([0.3, 0.1, 0.4, 0.2])#10
#         pmf = torch.tensor([0.1, 0.4, 0.2, 0.3])#11
        pmf = torch.tensor([0.4, 0.2, 0.3, 0.1])#12
#         pmf = torch.tensor([0.2, 0.3, 0.1, 0.4])#13
#         K=4
#         pmf = torch.tensor(torch.ones(K).float()/K)#14

        A1 = torch.distributions.categorical.Categorical(pmf)
        a1 = A1.sample(torch.Size([batch_size,1])).squeeze_(0)

#         print(a1.shape)
        data = torch.cat((c1, a1), 1).float()
#         print(data[:2])
        return torch.cat((c1, a1), 1).float()

    elif data == "cont_disc_n3":
        from scipy.stats import norm
        import torch
        from torch.distributions.normal import Normal
        from torch.distributions.bernoulli import Bernoulli
        mu_C1 = 0.0
        sig_C1 = 0.5

        C1 = Normal(torch.tensor([mu_C1]), torch.tensor([sig_C1]))
        c1 = C1.sample(sample_shape=torch.Size([batch_size]))  # normally distributed with loc=0 and scale=1

        p_A1 = norm.cdf(0.4*c1)
        p_A1 = torch.from_numpy(p_A1)

        A1 = Bernoulli(p_A1)
        a1 = A1.sample(sample_shape=torch.Size([1]))  # normally distributed with loc=0 and scale=1
        a1.squeeze_(0)
        
#         pmf = torch.tensor([0.4, 0.6])
#         pmf = torch.tensor([0.6, 0.4])
#         pmf = torch.tensor([0.3, 0.7])
#         pmf = torch.tensor([0.7, 0.3])
#         pmf = torch.tensor([0.3, 0.5, 0.2])
#         pmf = torch.tensor([0.3, 0.1, 0.4, 0.2])
#         K=4
#         pmf = torch.tensor(torch.ones(K).float()/K)
#         A1 = torch.distributions.categorical.Categorical(pmf)
#         a1 = A1.sample(torch.Size([batch_size,1])).squeeze_(0)
#         print(a1.shape)
        data = torch.cat((c1, a1), 1).float()
#         print(data[:2])
        return torch.cat((c1, a1), 1).float()

    else:
        return inf_train_gen("8gaussians", rng, batch_size)


