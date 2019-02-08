import zounds
import os


class FileSystemDataset(object):
    """
    A class conforming to the zounds dataset API that recursively yields wav
    files from a directory on-disk.  Absolute urls are generated for each file
    from a user-provided base_url and the file path and name.
    """

    EXTENSIONS = set(['.wav'])

    def __init__(self, path, base_url):
        super(FileSystemDataset, self).__init__()
        self.path = path
        self.base_url = base_url

    def _iter_files(self):
        """
        Recursively find all files with an allowed extension
        """
        for base_path, _, filenames in os.walk(self.path):
            for filename in filenames:
                _, extension = os.path.splitext(filename)
                if extension in self.EXTENSIONS:
                    yield base_path, filename

    def __iter__(self):
        """
        Recursively transform all wav files from a directory into zounds
        AudioMetaData instances
        """
        for base_path, filename in self._iter_files():
            full_path = os.path.join(base_path, filename)
            relative = os.path.relpath(full_path, self.path)
            url = os.path.join(self.base_url, relative)

            try:
                samples = zounds.AudioSamples.from_file(full_path)
            except (ValueError, RuntimeError):
                # the file may be corrupted.  skip it
                continue

            uri = zounds.datasets.PreDownload(samples.encode().read(), url)
            yield zounds.AudioMetaData(
                uri=uri,
                samplerate=int(samples.samplerate))


datasets = [

    FileSystemDataset('/hdd/musicnet/train_data/', 'https://musicnet.org/'),

    FileSystemDataset('/hdd/freesound', 'https://freesound.org'),

    zounds.PhatDrumLoops(),

    # Classical
    zounds.InternetArchive('AOC11B'),
    zounds.InternetArchive('CHOPINBallades-NEWTRANSFER'),
    zounds.InternetArchive('JohnCage'),
    zounds.InternetArchive('beethoven_ingigong_850'),
    zounds.InternetArchive('jamendo-086440'),
    zounds.InternetArchive('The_Four_Seasons_Vivaldi-10361'),

    # Pop
    zounds.InternetArchive('02.LostInTheShadowsLouGramm'),
    zounds.InternetArchive('08Scandalous'),
    zounds.InternetArchive('09.JoyceKennedyDidntITellYou'),
    zounds.InternetArchive('02.InThisCountryRobinZander'),
    zounds.InternetArchive('PeterGabrielOutOut'),
    zounds.InternetArchive('07.SpeedOfLightJoeSatriani'),

    # Jazz
    zounds.InternetArchive('Free_20s_Jazz_Collection'),
    zounds.InternetArchive('0BlueTrain'),
    zounds.InternetArchive('JohnColtrane-GiantSteps'),
    zounds.InternetArchive(
        'cd_john-coltrane-and-the-jazz-giants_john-coltrane-and-the-jazz-giants-john-col'),
    zounds.InternetArchive('cd_coltraneprestige-7105_john-coltrane'),
    zounds.InternetArchive(
        'cd_thelonious-monk-with-john-coltrane_thelonious-monk-john-coltrane'),

    # Hip Hop
    zounds.InternetArchive('LucaBrasi2'),
    zounds.InternetArchive('Chance_The_Rapper_-_Coloring_Book'),
    zounds.InternetArchive('Chance_The_Rapper_-_Acid_Rap-2013'),
    zounds.InternetArchive('Kevin_Gates_-_By_Any_Means-2014'),
    zounds.InternetArchive('Lil_Wayne_-_Public_Enemy'),
    zounds.InternetArchive('Chance_The_Rapper_-_Good_Enough'),

    # Speech
    zounds.InternetArchive('Greatest_Speeches_of_the_20th_Century'),
    zounds.InternetArchive(
        'cd_great-speeches-and-soliloquies_william-shakespeare'),
    zounds.InternetArchive('The_Speeches-8291'),
    zounds.InternetArchive('RasKitchen'),

    # Electronic
    zounds.InternetArchive('rome_sample_pack'),
    zounds.InternetArchive('CityGenetic'),
    zounds.InternetArchive('SvenMeyer-KickSamplePack'),
    zounds.InternetArchive('jamendo-046316'),
    zounds.InternetArchive('jamendo-079926'),
    zounds.InternetArchive('jamendo-069115'),
    zounds.InternetArchive('SampleScienceToyKeyboardSamples'),
    zounds.InternetArchive('jamendo-071495'),
    zounds.InternetArchive('HgfortuneTheTygerSynth'),
    zounds.InternetArchive('mellow-jeremy-synth-technology'),
    zounds.InternetArchive('RandomSynth2'),
    zounds.InternetArchive('Mc303Synth'),
    zounds.InternetArchive('HalloweenStickSynthRaver'),

    # Nintendo
    zounds.InternetArchive('CastlevaniaNESMusicStage10WalkingOnTheEdge'),
    zounds.InternetArchive(
        'BloodyTearsSSHRemixCastlevaniaIISimonsQuestMusicExtended'),
    zounds.InternetArchive(
        'CastlevaniaIIIDraculasCurseNESMusicEnterNameEpitaph'),
    zounds.InternetArchive('SuperMarioBros3NESMusicWorldMap6'),
    zounds.InternetArchive('SuperMarioBrosNESMusicHurriedOverworld'),
    zounds.InternetArchive('AdventuresOfGilligansIslandTheSoundtrack1NESMusic'),
    zounds.InternetArchive('SuperMarioWorldSNESMusicUndergroundThemeYoshi'),
]

dataset = zounds.CompositeDataset(*datasets)
