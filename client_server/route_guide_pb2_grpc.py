# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import route_guide_pb2 as route__guide__pb2


class RouteGuideStub(object):
  """Interface exported by the server.
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.GetFeature = channel.unary_unary(
        '/routeguide.RouteGuide/GetFeature',
        request_serializer=route__guide__pb2.Vector.SerializeToString,
        response_deserializer=route__guide__pb2.Vector.FromString,
        )


class RouteGuideServicer(object):
  """Interface exported by the server.
  """

  def GetFeature(self, request, context):
    """A simple RPC.

    Obtains the feature at a given position.

    A feature with an empty name is returned if there's no feature at the given
    position.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_RouteGuideServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'GetFeature': grpc.unary_unary_rpc_method_handler(
          servicer.GetFeature,
          request_deserializer=route__guide__pb2.Vector.FromString,
          response_serializer=route__guide__pb2.Vector.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'routeguide.RouteGuide', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
