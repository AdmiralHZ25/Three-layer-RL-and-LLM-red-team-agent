"""Facts found in the output of a Caldera link, and the goals and stage requirements counted from them.

A fact is a tuple (kind, value):
  ('ip', '192.168.56.10')  IPv4 address. Loopback, unspecified, multicast and reserved addresses (this includes
                           netmasks such as 255.255.255.0 and the broadcast address) are left out.
  ('host', 'DESKTOP-XYZ')  Windows host name (NetBIOS, at most 15 characters), upper case.
  ('port', '445')          TCP or UDP port that is listening or open, as text.
  ('user', 'alice')        account name, lower case. From DOMAIN\\user lines and USERNAME= lines.

extract_facts() uses simple regular expressions. It returns a sorted list without duplicates, so the same output always
gives the same list. The output comes from the target, so it is data: facts are only counted here. They are never
written into a prompt as an instruction and never run.

A goal, or a requirement of a stage, is a pair (name, n). It is reached when at least n different values of its kind are
known. GOAL_KINDS gives the kind for each name: ('host_lain', 2) means at least two different host names.
Known limitation: the output does not say which host is the agent's own, so the agent's own host counts too.
"""
import ipaddress
import re

# goal or requirement name -> fact kind that it counts
GOAL_KINDS = {'host_lain': 'host', 'ip': 'ip', 'port': 'port', 'user': 'user'}

_IP = re.compile(r'(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?![\d.]*\d)')
_HOST_FIELD = re.compile(r'host[ \t]?name[ \t.]*:[ \t]*([A-Za-z0-9][A-Za-z0-9-]{0,14})(?![A-Za-z0-9-])', re.I)
_COMPUTERNAME = re.compile(r'\bCOMPUTERNAME=([A-Za-z0-9][A-Za-z0-9-]{0,14})(?![A-Za-z0-9-])', re.I)
_UNC_HOST = re.compile(r'\\\\([A-Za-z0-9][A-Za-z0-9-]{0,14})(?=[\\\s]|$)', re.M)
_NETSTAT_PORT = re.compile(r'^[ \t]*(?:TCP|UDP)[ \t]+\S+:(\d{1,5})[ \t]', re.M | re.I)
_NMAP_PORT = re.compile(r'^[ \t]*(\d{1,5})/(?:tcp|udp)[ \t]+open\b', re.M | re.I)
_DOMAIN_USER = re.compile(r'^[ \t]*[\w .-]+\\([\w.$-]+)[ \t]*$', re.M)
_USERNAME_ENV = re.compile(r'\bUSERNAME=([\w.$-]+)', re.I)


def _norm(kind, value):
    """Host names and account names are not case sensitive on Windows, so they are compared in one case."""
    value = str(value).strip()
    if kind == 'host':
        return value.upper()
    if kind == 'user':
        return value.lower()
    return value


def _ips(text):
    out = set()
    for m in _IP.finditer(text):
        try:
            addr = ipaddress.IPv4Address(m.group(1))
        except ValueError:  # an octet above 255, or a leading zero
            continue
        if addr.is_loopback or addr.is_unspecified or addr.is_multicast or addr.is_reserved:
            continue
        out.add(('ip', str(addr)))
    return out


def _hosts(text):
    names = [m.group(1) for m in _HOST_FIELD.finditer(text)]
    names += [m.group(1) for m in _COMPUTERNAME.finditer(text)]
    names += [m.group(1) for m in _UNC_HOST.finditer(text)]
    return {('host', _norm('host', n)) for n in names}


def _ports(text):
    out = set()
    for m in list(_NETSTAT_PORT.finditer(text)) + list(_NMAP_PORT.finditer(text)):
        port = int(m.group(1))
        if 1 <= port <= 65535:
            out.add(('port', str(port)))
    return out


def _users(text):
    names = [m.group(1) for m in _DOMAIN_USER.finditer(text)]
    names += [m.group(1) for m in _USERNAME_ENV.finditer(text)]
    return {('user', _norm('user', n)) for n in names}


def extract_facts(output_text):
    """Facts found in one output text, as a sorted list of unique (kind, value) tuples. None or '' gives []."""
    text = output_text or ''
    return sorted(_ips(text) | _hosts(text) | _ports(text) | _users(text))


def check_need(need):
    """Return (name, n) when it is a goal or requirement that can be counted. Raises ValueError otherwise, so a
    typing error stops the run before anything is sent."""
    try:
        name, n = need
    except (TypeError, ValueError):
        raise ValueError(f'a goal or requirement is a pair (name, n), got {need!r}') from None
    if not isinstance(name, str) or name not in GOAL_KINDS:
        raise ValueError(f'unknown goal or requirement {name!r}; use one of {sorted(GOAL_KINDS)}')
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f'the count in {need!r} must be a whole number of at least 1')
    return name, n


def goal_reached(facts, goal):
    """True when at least n different values of the goal's kind are in facts (an iterable of (kind, value))."""
    name, n = check_need(goal)
    kind = GOAL_KINDS[name]
    values = {_norm(k, v) for k, v in facts if k == kind}
    return len(values) >= n
