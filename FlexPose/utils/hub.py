import torch


def load_FlexPose(model_dir=None):
    print('Loading FlexPose parameters ...')
    chk = torch.hub.load_state_dict_from_url(
        'http://www.knightofnight.com/upload/data/FlexPose/FlexPose_param.chk',
        model_dir=model_dir,
        map_location='cpu',
        progress=True
    )
    return chk


def load_pretrained_protein_encoder(model_dir=None):
    print('Loading pre-trained protein encoder ...')
    chk = torch.hub.load_state_dict_from_url(
        'http://www.knightofnight.com/upload/data/FlexPose/FlexPose_pretrained_protein_encoder.chk',
        model_dir=model_dir,
        map_location='cpu',
        progress=True
    )

    return chk


def load_pretrained_ligand_encoder(model_dir=None):
    print('Loading pre-trained ligand encoder ...')
    chk = torch.hub.load_state_dict_from_url(
        'http://www.knightofnight.com/upload/data/FlexPose/FlexPose_pretrained_ligand_encoder.chk',
        model_dir=model_dir,
        map_location='cpu',
        progress=True
    )

    return chk


if __name__ == '__main__':
    chk = load_FlexPose('test')
    print(chk.keys())

    chk = load_pretrained_protein_encoder('test')
    print(chk.keys())

    chk = load_pretrained_ligand_encoder('test')
    print(chk.keys())


