import torch



def means_circle(k=8):
    p = 3.14159265359
    t = torch.linspace(0, 2 * p - (2 * p / k), k)
    m = torch.cat((torch.sin(t).view(-1, 1),
                   torch.cos(t).view(-1, 1)), 1)
    print(m)

if __name__ == '__main__':
    print(means_circle(8))