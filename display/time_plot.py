import matplotlib.pyplot as plt

headers = ['P [GeV]', 'K [GeV]', 'T [GeV]', 't', 'gamma']


def visualize_test(P, K, T, model, N=50, df_truth=None):
    # file = 'datasets/combined_rates.csv'
    # df = pd.read_csv(file)

    # df = df['T' == 0.3]
    # print(df.shape)
    # post_results = model(test[0])
    # print(((post_results - init_results)**2).mean().item())

    t = torch.linspace(0, 12, N)

    P = torch.ones(N)*P
    K = torch.ones(N)*K
    T = torch.ones(N)*T

    stack = torch.stack([P, K, T, t]).T
    model.eval()
    out = model(stack)

    plt.plot(t.detach().numpy(), out.detach().numpy())
    plt.show()