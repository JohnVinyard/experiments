import zounds


class Dataset(object):
    """
    A small dataset containing varied types of audio:
        - speech
        - hip hop
        - classical piano
        - drum kit
    """

    def __init__(self):
        super(Dataset, self).__init__()

    def __iter__(self):
        bach = zounds.InternetArchive('AOC11B')
        kevin_gates = zounds.InternetArchive('Kevin_Gates_-_By_Any_Means-2014')
        drums = zounds.PhatDrumLoops()
        speech = zounds.InternetArchive('Greatest_Speeches_of_the_20th_Century')
        yield iter(bach).next()
        yield iter(kevin_gates).next()
        yield iter(drums).next()
        yield iter(speech).next()
