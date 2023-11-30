from FlexPose.utils.prediction import predict as predict_by_FlexPose

predict_by_FlexPose(
    protein='./FlexPose/example/4r6e/4r6e_protein.pdb',               # a protein path, or a list of paths
    ligand='./FlexPose/example/4r6e/4r6e_ligand.mol2',                # a ligand path (or SMILES), or a list of paths (or SMILES)
    ref_pocket_center='./FlexPose/example/4r6e/4r6e_ligand.mol2',     # a file for pocket center prediction, e.g. predictions from Fpocket
    # batch_csv='./FlexPose/example/example_input.csv',               # batch prediction

    device='cuda:0',                                                  # device
    structure_output_path='./structure_output',                       # structure output
    output_result_path='./output.csv',                                # record output
)


