"""
July 27th, 2022
CDMX
M. Sagols

Trading in Interactive Brokers Project
Program to establish the connection to the database server
"""

import os
import socket
import psycopg2
import sshtunnel
from sshtunnel import SSHTunnelForwarder


CONNECTION = None


def connection_tunnel(database='IBdb', logger=None):
    """
    This function returns a connection to Postgresql in the '10.244.109.140'
    server.

    Parameters
    ----------
    database : str
        Name of the database.
    logger :
        Program's logger.
    """
    global CONNECTION

    sshtunnel.SSH_TIMEOUT = 5
    sshtunnel.TUNNEL_TIMEOUT = 30
    if CONNECTION is not None:
        return CONNECTION
    host_name = socket.gethostname()
    pass_ = 'a33378l#222e98789b9909r004i7j654es@@'
    server = None
    if host_name != 'aishia':
        server = SSHTunnelForwarder((os.environ['aishia_ip'],
                                     int(os.environ['aishia_port'])),
                                    ssh_password=pass_,
                                    ssh_username=os.environ['aishia_u'],
                                    allow_agent=False,
                                    remote_bind_address=('127.0.0.1', 5432))
        server.start()
    params = {
        'database': database,
        'user': os.environ['aishia_u'],
        'password': os.environ['aishia_p'],
        'host': 'localhost',
        'port': server.local_bind_port if host_name != 'aishia' else 5432
    }
    """
    YOSHUA
    Necesario para conectarse a copia de IBdb para hacer pruebas
    """
    # params = {
    #     'host': 'localhost',
    #     'database': 'ibdb_local',
    #     'user': 'postgres',
    #     'password': 'postgrespass'
    # }
    try:
        CONNECTION = psycopg2.connect(**params)
    except psycopg2.OperationalError as error:
        if logger is not None:
            logger.critical(error)
        raise psycopg2.OperationalError(error)
    return CONNECTION


if __name__ == "__main__":
    conn = connection_tunnel()
    cur = conn.cursor()
    cur.execute("select * from accounts")
    xd = cur.fetchall()
    print(xd)
