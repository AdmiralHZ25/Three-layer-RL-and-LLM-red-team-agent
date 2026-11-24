"""Mapping: MITRE ATT&CK technique of a Caldera ability -> one CAGE-4 action class.

Design note: each row is a judgment call; ATT&CK and CAGE-4 do not align one-to-one. A technique maps to
the CAGE-4 class that best describes its strategic role in the kill chain. A candidate without a row is
not ranked by the RL agent; the LLM can still choose it. Edit the table, then run tests/test_caldera_runner.py.

Techniques ordered by frequency in actual Caldera runs (from runs/caldera/*/steps.jsonl, 2026-11-19).
"""

# Same names and order as red.simulator.cage4.NAMES (the test suite checks that they match).
CAGE_CLASSES = ('AggressiveServiceDiscovery', 'DegradeServices', 'DiscoverDeception',
                'DiscoverRemoteSystems', 'ExploitRemoteService', 'Impact',
                'PrivilegeEscalate', 'Sleep', 'StealthServiceDiscovery', 'Withdraw')

TECHNIQUE_TO_CLASS = {
    # --- Exploitation / remote code execution ---
    'T1047': 'ExploitRemoteService',      # WMI (most common in runs: 19)
    'T1059': 'ExploitRemoteService',      # Command and Scripting Interpreter (12)
    'T1021': 'ExploitRemoteService',      # Remote Services: SSH, RDP, SMB (2)
    'T1078': 'ExploitRemoteService',      # Valid Accounts — using creds to access systems
    'T1106': 'ExploitRemoteService',      # Native API — syscall-level code execution (4)
    'T1204': 'ExploitRemoteService',      # User Execution (1)
    'T1210': 'ExploitRemoteService',      # Exploitation of Remote Services
    'T1505': 'ExploitRemoteService',      # Server Software Component: web shells (2)
    'T1550': 'ExploitRemoteService',      # Use Alternate Authentication Material: PtH (1)
    'T1569': 'ExploitRemoteService',      # System Services — service-based execution (3)

    # --- Impact ---
    'T1490': 'Impact',                    # Inhibit System Recovery: VSS, backup (10)
    'T1485': 'Impact',                    # Data Destruction (2)
    'T1486': 'Impact',                    # Data Encrypted for Impact
    'T1496': 'Impact',                    # Resource Hijacking (1)
    'T1074': 'Impact',                    # Data Staged — collection for exfil (1)
    'T1119': 'Impact',                    # Automated Collection (1)

    # --- Degradation ---
    'T1489': 'DegradeServices',           # Service Stop (5)
    'T1499': 'DegradeServices',           # Endpoint Denial of Service
    'T1529': 'DegradeServices',           # System Shutdown/Reboot

    # --- Defense evasion → DiscoverDeception (blending in, avoiding detection) ---
    'T1218': 'DiscoverDeception',         # System Binary Proxy Execution: LOLBins (9)
    'T1036': 'DiscoverDeception',         # Masquerading (2)
    'T1070': 'DiscoverDeception',         # Indicator Removal (1)
    'T1112': 'DiscoverDeception',         # Modify Registry — evasion/persistence (2)
    'T1564': 'DiscoverDeception',         # Hide Artifacts (2)
    'T1027': 'DiscoverDeception',         # Obfuscated Files or Information

    # --- Privilege escalation ---
    'T1003': 'PrivilegeEscalate',         # OS Credential Dumping (5)
    'T1068': 'PrivilegeEscalate',         # Exploitation for Privilege Escalation (2)
    'T1098': 'PrivilegeEscalate',         # Account Manipulation
    'T1110': 'PrivilegeEscalate',         # Brute Force — credential access → privesc (2)
    'T1134': 'PrivilegeEscalate',         # Access Token Manipulation (3)
    'T1136': 'PrivilegeEscalate',         # Create Account (2)
    'T1543': 'PrivilegeEscalate',         # Create or Modify System Process (3)
    'T1548': 'PrivilegeEscalate',         # Abuse Elevation Control Mechanism (1)
    'T1552': 'PrivilegeEscalate',         # Unsecured Credentials
    'T1555': 'PrivilegeEscalate',         # Credentials from Password Stores (1)
    'T1558': 'PrivilegeEscalate',         # Steal or Forge Kerberos Tickets (2)
    'T1574': 'PrivilegeEscalate',         # Hijack Execution Flow: DLL hijack (2)

    # --- Discovery → quiet enumeration ---
    'T1007': 'StealthServiceDiscovery',   # System Service Discovery (5)
    'T1057': 'StealthServiceDiscovery',   # Process Discovery (7)
    'T1010': 'StealthServiceDiscovery',   # Application Window Discovery (1)
    'T1016': 'StealthServiceDiscovery',   # System Network Configuration Discovery
    'T1018': 'DiscoverRemoteSystems',     # Remote System Discovery (2)
    'T1033': 'StealthServiceDiscovery',   # System Owner/User Discovery
    'T1046': 'StealthServiceDiscovery',   # Network Service Discovery
    'T1049': 'StealthServiceDiscovery',   # System Network Connections Discovery
    'T1069': 'StealthServiceDiscovery',   # Permission Groups Discovery (2)
    'T1082': 'StealthServiceDiscovery',   # System Information Discovery (1)
    'T1083': 'StealthServiceDiscovery',   # File and Directory Discovery
    'T1087': 'StealthServiceDiscovery',   # Account Discovery
    'T1120': 'StealthServiceDiscovery',   # Peripheral Device Discovery (2)
    'T1135': 'DiscoverRemoteSystems',     # Network Share Discovery
    'T1482': 'DiscoverRemoteSystems',     # Domain Trust Discovery (1)
    'T1518': 'StealthServiceDiscovery',   # Software Discovery
    'T1685': 'StealthServiceDiscovery',   # CAGE-4 internal technique (3) — mapped as quiet discovery
}


def cage_class(technique_id):
    """CAGE-4 class for a technique id such as 'T1046' or 'T1046.001'. None when there is no row."""
    return TECHNIQUE_TO_CLASS.get((technique_id or '').split('.')[0].upper())
