import pandapower as pp
import numpy as np
import torch
import networkx as nx
import pandas as pd

def example_net(size):
    """
    Example of simple net on 3 phases
    """
    net = pp.create_empty_network(sn_mva = 100)

    bus_list = []

    busnr1 = pp.create_bus(net, name="bus1", vn_kv=10., geodata=(0, 0))

    for i in range(size):
        bus_list.append(pp.create_bus(net, name="bus"+str(i+2), vn_kv=0.4, geodata=(i+1, 0)))

    pp.create_ext_grid(net, busnr1, vm_pu= 1.0, s_sc_max_mva=5000, rx_max=0.1, r0x0_max= 0.1, x0x_max=1.0)

    pp.create_transformer_from_parameters(net, busnr1, bus_list[0], sn_mva=0.63,
                                      vn_hv_kv=10., vn_lv_kv=0.4,
                                      vkr_percent=0.1, vk_percent=6,
                                      vk0_percent=6, vkr0_percent=0.78125,
                                      mag0_percent=100, mag0_rx=0.,
                                      pfe_kw=0.1, i0_percent=0.1,
                                      vector_group="Dyn", shift_degree=150,
                                      si0_hv_partial=0.9)

    for i in range(size-1):
        pp.create_line_from_parameters(net, bus_list[i], bus_list[i+1], length_km=0.1, r0_ohm_per_km=0.0848,
                               x0_ohm_per_km=0.4649556, c0_nf_per_km=230.6,
                               max_i_ka=0.963, r_ohm_per_km=0.0212,
                               x_ohm_per_km=0.1162389, c_nf_per_km= 230)

    if size%3 == 0:
        for i in range(size):
            pp.create_asymmetric_load(net, bus_list[i], p_a_mw=0.003)

    elif size%3 == 1:
        for i in range(size-2):
            pp.create_asymmetric_load(net, bus_list[i], p_a_mw=0.003)
        pp.create_asymmetric_load(net, bus_list[-1], p_b_mw=0.002)
        pp.create_asymmetric_load(net, bus_list[-2], p_c_mw=0.001)

    else:
        for i in range(size-3):
            pp.create_asymmetric_load(net, bus_list[i], p_a_mw=0.003)
        pp.create_asymmetric_load(net, bus_list[-1], p_a_mw=0.001)
        pp.create_asymmetric_load(net, bus_list[-2], p_b_mw=0.001)
        pp.create_asymmetric_load(net, bus_list[-3], p_c_mw=0.001)

    pp.create_std_type(net, {"r0_ohm_per_km": 0.0848, "x0_ohm_per_km": 0.4649556, "c0_nf_per_km":\
    230.6,"max_i_ka": 0.963, "r_ohm_per_km": 0.0212, "x_ohm_per_km": 0.1162389,
             "c_nf_per_km":  230}, name="example_type")

    pp.add_zero_impedance_parameters(net)


    return net

def get_feature_adjacency_loads(net: pp.pandapowerNet, mask=None):
    """
    Returns the feature matrix and adjacency matrix of the network for its asymmetric loads.
    """

    assert type(net) == pp.pandapowerNet

    df = net.asymmetric_load
    df.sort_values(by='bus', inplace=True)
    df.reset_index(drop=True, inplace=True)

    mask = np.array(mask)
    assert len(mask) == len(df)
    assert mask.sum() > 0

    adj = nx.adjacency_matrix(pp.topology.create_nxgraph(net, respect_switches=False)).todense()
    adj = adj[:-1, :-1]
    adj = torch.tensor(adj, dtype=torch.float32)
    feature = np.zeros((df.shape[0], 4))
    for i in range(df.shape[0]):
        feature[i, 0] = 1 if df.p_a_mw.iloc[i] > 0 else 0
        feature[i, 1] = 1 if df.p_b_mw.iloc[i] > 0 else 0
        feature[i, 2] = 1 if df.p_c_mw.iloc[i] > 0 else 0

    feature = torch.tensor(feature, dtype=torch.float32)

    if mask is not None:
        # Mask out some of the features (e.g. mask = [0, 1, 1, 0, 0, 1, 1, 0, 0, 0])
        mask = torch.tensor(mask, dtype=torch.float32).view(-1, 1)
        feature = (feature * mask)

    feature[:, 3] = torch.tensor(df.p_a_mw + df.p_b_mw + df.p_c_mw)

    return feature, adj

def create_loads(size, ts):
    df = pd.DataFrame()
    if size%3 == 0:
        for i in range(size):
            name = 'load_'+ str(i)
            df[name] = np.random.rand(ts) * 0.0001
            df[name] += 0.003

    elif size%3 == 1:
        for i in range(size):
            name = 'load_'+ str(i)
            df[name] = np.random.rand(ts) * 0.0001

            if i == size-1:
                df[name] += 0.002
            elif i == size-2:
                df[name] += 0.001
            else:
                df[name] += 0.003

    else:
        for i in range(size):
            name = 'load_'+ str(i)
            df[name] = np.random.rand(ts) * 0.0001

            if i == size-1 or i == size-2 or i==size-3:
                df[name] += 0.001
            else:
                df[name] += 0.003
    if ts > 10:
        df.to_csv('./loads/loads_big_'+str(size)+'.csv', index=False)
    else:
        df.to_csv('./loads/loads_small_'+str(size)+'.csv', index=False)

    return df

if __name__ == '__main__':
    tss = [10,100]
    sizes = [5,6,7]
    for ts in tss:
        for size in sizes:
            create_loads(size, ts)