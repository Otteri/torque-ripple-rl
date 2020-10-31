# expects: ten batch 
def test1(input, target):
    print("Input shape:", input.shape)
    print("Target shape:", target.shape)

    # if(input.shape == torch.Size(10, 2, 999)):
    #     print("Shape is correct!")


    # Check signal first value
    assert (target[0, 1, 0] == input[0, 1, 1]), "First item should be same"

    assert (target[9, 1, 0] == input[9, 1, 1]), "First item should be same"

    # Check signal last value
    assert (target[0, 1, -2] == input[0, 1, -1]), "Last item should be same"

    assert (target[9, 1, -2] == input[9, 1, -1]), "Last item should be same"