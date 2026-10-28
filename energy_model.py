from dataclasses import dataclass, asdict

E_MULT_PJ = 3.7
E_ADD_PJ = 0.9
E_MAC_PJ = E_MULT_PJ + E_ADD_PJ # 4.6 pJ


@dataclass(frozen=True)
class Assumptions:
    e_mac: float = E_MAC_PJ
    e_add: float = E_ADD_PJ
    e_update: float = E_MAC_PJ
    e_compare: float = E_ADD_PJ
    binarisation_shared: bool = True # set to False only if the SNN gets its own encoder 
    act_cost_in_macs: float = 0.0
    ann_binary_inputs_as_adds: bool = False  # if True, ANN synapses coming from the (binary) input layer cost an add instead of a MAC
    # R-STDP. every operation counted in plasticity_updates is one multiply-accumulate: an eligibility fade (e*decay),
    # a potentiation/depression (e +- A*trace) or a weight update (w + (lr*dopamine)*e, lr*dopamine shared per frame)
    e_plasticity: float = E_MAC_PJ
    # a trace fade (trace*decay) is one multiplication, priced as a MAC like the membrane update (conservative)
    e_trace: float = E_MAC_PJ

    def as_dict(self):
        return asdict(self)


DEFAULT = Assumptions()


def ann_energy(frames, n_conn, n_nodes, n_conn_from_input=0, a: Assumptions = DEFAULT):
    """Dense recurrent ANN (neat-python RecurrentNetwork): every enabled connection is a MAC every frame,
    every node adds its bias and evaluates its activation every frame."""
    
    if a.ann_binary_inputs_as_adds:
        mac = frames * ((n_conn - n_conn_from_input) * a.e_mac + n_conn_from_input * a.e_add)
    else:
        mac = frames * n_conn * a.e_mac
    bias = frames * n_nodes * a.e_add
    act = frames * n_nodes * a.act_cost_in_macs * a.e_mac
    return {"mac_pj": mac, "bias_pj": bias, "act_pj": act, "total_pj": mac + bias + act}


def snn_energy(frames, synops, neuron_updates, n_inputs, plasticity_updates=0, a: Assumptions = DEFAULT,
               trace_updates=0):
    """Event-driven SNN: one accumulate per spike delivered through a synapse, one membrane update per neuron
    per frame, one threshold comparison per input pixel per frame (encoding). With R-STDP, also the synapse work
    (eligibility + weight updates) and the trace fades, counted only when they actually happen."""

    syn = synops * a.e_add
    upd = neuron_updates * a.e_update
    shared = frames * n_inputs * a.e_compare
    enc = 0.0 if a.binarisation_shared else shared
    pla = plasticity_updates * a.e_plasticity
    tra = trace_updates * a.e_trace
    return {"synop_pj": syn, "update_pj": upd, "encoding_pj": enc, "plasticity_pj": pla, "trace_pj": tra,
            "shared_preproc_pj": shared,   # informative only, not in total when binarisation_shared
            "total_pj": syn + upd + enc + pla + tra}