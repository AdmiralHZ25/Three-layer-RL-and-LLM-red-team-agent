"""Draft mapping: MITRE ATT&CK technique of a Caldera ability -> one CAGE-4 action class.

REVIEW_REQUIRED: this table is a design choice, not a validated result. A candidate without a row is not ranked
by the RL agent; the LLM can still choose it. Edit the table, then run tests/test_caldera_runner.py.
"""

# Same names and order as red.simulator.cage4.NAMES (the test suite checks that they match).
CAGE_CLASSES = ('AggressiveServiceDiscovery', 'DegradeServices', 'DiscoverDeception',
                'DiscoverRemoteSystems', 'ExploitRemoteService', 'Impact',
                'PrivilegeEscalate', 'Sleep', 'StealthServiceDiscovery', 'Withdraw')

TECHNIQUE_TO_CLASS = {
    'T1018': 'DiscoverRemoteSystems',    # Remote System Discovery
    'T1046': 'StealthServiceDiscovery',  # Network Service Discovery (draft: the quieter class)
    'T1210': 'ExploitRemoteService',     # Exploitation of Remote Services
    'T1068': 'PrivilegeEscalate',        # Exploitation for Privilege Escalation
    'T1548': 'PrivilegeEscalate',        # Abuse Elevation Control Mechanism
    'T1489': 'DegradeServices',          # Service Stop
    'T1485': 'Impact',                   # Data Destruction
    'T1486': 'Impact',                   # Data Encrypted for Impact
}


def cage_class(technique_id):
    """CAGE-4 class for a technique id such as 'T1046' or 'T1046.001'. None when there is no row."""
    return TECHNIQUE_TO_CLASS.get((technique_id or '').split('.')[0].upper())
